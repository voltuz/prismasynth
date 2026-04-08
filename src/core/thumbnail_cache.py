import logging
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from core.proxy_cache import ProxyManager
from core.timeline import TimelineModel
from core.video_source import VideoSource

logger = logging.getLogger(__name__)

THUMB_WIDTH = 192
THUMB_HEIGHT = 108
THUMB_FRAME_SIZE = THUMB_WIDTH * THUMB_HEIGHT * 3
_MAX_WORKERS = 4


class ThumbnailCache(QObject):
    """Viewport-only thumbnail generation with playhead-distance priority.

    Only generates thumbnails for clips currently visible in the viewport.
    4 parallel ffmpeg -ss seeks for throughput (~38ms/frame effective).
    Re-checks priority between every submission so viewport changes
    take effect within one seek (~100ms).
    """

    thumbnail_ready = Signal(str, str, QImage)  # clip_id, "first"|"last", qimage

    def __init__(self, timeline: TimelineModel,
                 sources: Dict[str, VideoSource],
                 proxy_manager: ProxyManager = None, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._proxy_manager = proxy_manager
        self._mem_cache: Dict[str, QImage] = {}
        self._lq_emitted: Set[str] = set()  # cache keys where LQ placeholder was sent
        self._stopped = False
        self._coord_thread: Optional[threading.Thread] = None

        # Pause support (scrubbing priority)
        self._pause_event = threading.Event()
        self._pause_event.set()

        # Priority management — updated from main thread on scroll
        self._priority_clip_ids: Set[str] = set()
        self._playhead_frame: int = 0
        self._priority_lock = threading.Lock()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def generate_all(self, priority_clip_ids: Set[str] = None,
                     playhead_frame: int = 0):
        """Start thumbnail generation. priority_clip_ids are generated first."""
        if self._coord_thread and self._coord_thread.is_alive():
            if priority_clip_ids is not None:
                self.reprioritize(priority_clip_ids, playhead_frame)
            return
        self._stopped = False
        with self._priority_lock:
            self._priority_clip_ids = set(priority_clip_ids or [])
            self._playhead_frame = playhead_frame
        self._coord_thread = threading.Thread(target=self._coordinate, daemon=True)
        self._coord_thread.start()

    def reprioritize(self, visible_clip_ids: Set[str],
                     playhead_frame: int = 0):
        """Called on scroll — next submission uses the new viewport."""
        with self._priority_lock:
            self._priority_clip_ids = set(visible_clip_ids)
            self._playhead_frame = playhead_frame

    def _coordinate(self):
        """Generate thumbnails for visible clips only, 4 parallel seeks,
        priority re-checked between every submission."""
        # Build clip lookup once
        clip_lookup: Dict[str, List[Tuple[str, int, str]]] = {}
        for clip in self._timeline.clips:
            if clip.is_gap:
                continue
            clip_lookup[clip.id] = [
                (clip.source_id, clip.source_in, "first"),
                (clip.source_id, clip.source_out, "last"),
            ]

        pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        # In-flight futures: future -> (source_id, frame_num, clip_id, position)
        in_flight: Dict[Future, Tuple[str, int, str, str]] = {}

        try:
            while not self._stopped:
                # Wait if paused (scrubbing)
                while not self._pause_event.wait(timeout=0.1):
                    if self._stopped:
                        return

                # Drain completed futures
                done = [f for f in in_flight if f.done()]
                for f in done:
                    source_id, frame_num, clip_id, position = in_flight.pop(f)
                    try:
                        qimage = f.result()
                    except Exception:
                        continue
                    if qimage is not None:
                        cache_key = f"{source_id}_{frame_num}"
                        self._mem_cache[cache_key] = qimage
                        self.thumbnail_ready.emit(clip_id, position, qimage)

                # Fill pool up to _MAX_WORKERS with highest-priority visible frames
                while len(in_flight) < _MAX_WORKERS and not self._stopped:
                    frame = self._pick_next(clip_lookup, in_flight)
                    if frame is None:
                        break  # Nothing to submit
                    source_id, frame_num, clip_id, position = frame
                    source = self._sources.get(source_id)
                    if source is None:
                        continue
                    fut = pool.submit(self._grab_frame, source, frame_num)
                    in_flight[fut] = (source_id, frame_num, clip_id, position)

                if not in_flight:
                    # Nothing in-flight, nothing to submit — idle
                    time.sleep(0.05)
                else:
                    # Brief sleep to avoid busy-spinning on future checks
                    time.sleep(0.01)

        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _pick_next(self, clip_lookup, in_flight):
        """Pick the highest-priority uncached visible frame not already in-flight."""
        with self._priority_lock:
            visible_ids = self._priority_clip_ids
            playhead = self._playhead_frame

        # Already in-flight frame keys
        in_flight_keys = {(v[0], v[1]) for v in in_flight.values()}

        # Collect uncached visible frames
        needed = []
        for clip_id in visible_ids:
            entries = clip_lookup.get(clip_id)
            if not entries:
                continue
            for source_id, frame_num, position in entries:
                key = (source_id, frame_num)
                if key in in_flight_keys:
                    continue
                cache_key = f"{source_id}_{frame_num}"
                if cache_key in self._mem_cache:
                    # Already cached (HQ) — emit and skip
                    self.thumbnail_ready.emit(
                        clip_id, position, self._mem_cache[cache_key])
                    continue
                # Emit LQ placeholder from proxy if available
                if (cache_key not in self._lq_emitted
                        and self._proxy_manager is not None):
                    proxy = self._proxy_manager.get_proxy(source_id)
                    if proxy and frame_num < proxy.n_frames:
                        lq_frame = proxy.get_frame(frame_num)
                        if lq_frame is not None:
                            thumb = cv2.resize(
                                lq_frame, (THUMB_WIDTH, THUMB_HEIGHT),
                                interpolation=cv2.INTER_NEAREST)
                            h, w, ch = thumb.shape
                            qimg = QImage(bytes(thumb.data), w, h, ch * w,
                                          QImage.Format.Format_RGB888).copy()
                            self.thumbnail_ready.emit(clip_id, position, qimg)
                            self._lq_emitted.add(cache_key)
                needed.append((source_id, frame_num, clip_id, position))

        if not needed:
            return None

        # Closest to playhead first
        needed.sort(key=lambda r: abs(r[1] - playhead))
        return needed[0]

    @staticmethod
    def _grab_frame(source: VideoSource,
                    frame_num: int) -> Optional[QImage]:
        """Grab one frame via ffmpeg -ss seek (~100ms)."""
        timestamp = frame_num / source.fps if source.fps > 0 else 0.0
        cmd = [
            "ffmpeg", "-nostdin", "-v", "quiet",
            "-ss", f"{timestamp:.4f}",
            "-i", source.file_path,
            "-frames:v", "1",
            "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            data, _ = proc.communicate(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            if proc:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            return None

        if len(data) != THUMB_FRAME_SIZE:
            return None

        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            THUMB_HEIGHT, THUMB_WIDTH, 3)
        h, w, ch = rgb.shape
        return QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()

    def stop(self):
        self._stopped = True
        self._pause_event.set()
        if self._coord_thread and self._coord_thread.is_alive():
            self._coord_thread.join(timeout=3.0)
        self._coord_thread = None

    def invalidate_source(self, source_id: str):
        """Remove all cached thumbnails for a source so they regenerate."""
        keys = [k for k in self._mem_cache if k.startswith(f"{source_id}_")]
        for k in keys:
            del self._mem_cache[k]

    def clear(self):
        self._mem_cache.clear()
