import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Set, Tuple

import av
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
_MAX_WORKERS = 6

# If the next target is within this many frames, keep decoding forward
# instead of re-seeking. Roughly 2x typical H.265 GOP size.
_SEQUENTIAL_THRESHOLD = 400


class ThumbnailCache(QObject):
    """Long-lived thumbnail generator. Created once, survives timeline edits.

    Uses a sweep strategy: from the playhead, sweeps right (forward) and left
    (backward) through sorted frame targets. Within each sweep, sequential
    decode avoids redundant re-seeks for nearby frames. Distant jumps fall
    back to independent seeks.
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
        self._lq_emitted: Set[str] = set()
        self._stopped = False
        self._coord_thread: Optional[threading.Thread] = None
        self._pool: Optional[ThreadPoolExecutor] = None

        # Pause support (scrubbing priority)
        self._pause_event = threading.Event()
        self._pause_event.set()

        # Clip lookup — rebuilt on notify_clips_changed()
        self._clip_lookup: Dict[str, List[Tuple[str, int, str]]] = {}
        self._lookup_lock = threading.Lock()

        # Priority management — updated from main thread on scroll
        self._priority_clip_ids: Set[str] = set()
        self._playhead_frame: int = 0
        self._priority_lock = threading.Lock()

        # Wake event — signals coordinator to re-check for work
        self._wake_event = threading.Event()

        # When False, coordinator still emits LQ proxy placeholders but skips
        # submitting expensive HQ ffmpeg/PyAV decode jobs.
        self._hq_enabled = True

        # Track live subprocesses so stop() can kill them immediately
        # instead of waiting for per-frame communicate() timeouts.
        self._active_procs: Set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()

    def set_hq_enabled(self, enabled: bool):
        """Toggle HQ thumbnail generation. LQ proxy placeholders keep emitting
        either way while the cache is running."""
        self._hq_enabled = enabled
        self._wake_event.set()

    def pause(self):
        self._pause_event.clear()
        # Kill in-flight ffmpeg subprocesses so single-frame grabs abort
        # immediately instead of holding CPU/disk through communicate().
        with self._procs_lock:
            procs = list(self._active_procs)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    def resume(self):
        self._pause_event.set()
        self._wake_event.set()

    def start(self, priority_clip_ids: Set[str] = None,
              playhead_frame: int = 0):
        """Start the persistent coordinator thread if not already running."""
        with self._priority_lock:
            self._priority_clip_ids = set(priority_clip_ids or [])
            self._playhead_frame = playhead_frame
        self._rebuild_clip_lookup()
        if self._coord_thread and self._coord_thread.is_alive():
            self._wake_event.set()
            return
        self._stopped = False
        self._pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        self._coord_thread = threading.Thread(target=self._coordinate, daemon=True)
        self._coord_thread.start()

    def notify_clips_changed(self):
        """Called when clips change (split, detection, import, delete).
        Rebuilds the clip lookup without destroying the cache or thread pool."""
        self._rebuild_clip_lookup()
        self._wake_event.set()

    def _rebuild_clip_lookup(self):
        lookup: Dict[str, List[Tuple[str, int, str]]] = {}
        for clip in self._timeline.clips:
            if clip.is_gap:
                continue
            lookup[clip.id] = [
                (clip.source_id, clip.source_in, "first"),
                (clip.source_id, clip.source_out, "last"),
            ]
        with self._lookup_lock:
            self._clip_lookup = lookup

    def reprioritize(self, visible_clip_ids: Set[str],
                     playhead_frame: int = 0):
        """Called on scroll — next submission uses the new viewport."""
        with self._priority_lock:
            self._priority_clip_ids = set(visible_clip_ids)
            self._playhead_frame = playhead_frame
        self._wake_event.set()

    def _coordinate(self):
        """Persistent coordinator — plans sweep batches, dispatches to pool."""
        in_flight: Dict[Future, List[Tuple[str, int, str, str]]] = {}

        try:
            while not self._stopped:
                # Wait if paused (scrubbing)
                while not self._pause_event.wait(timeout=0.1):
                    if self._stopped:
                        return

                # Drain completed futures and emit results
                done = [f for f in in_flight if f.done()]
                for f in done:
                    entries = in_flight.pop(f)
                    try:
                        results = f.result()  # list of (frame_num, QImage)
                    except Exception:
                        continue
                    if results is None:
                        continue
                    # Map results back to clip_id/position for emission
                    for frame_num, qimage in results:
                        for source_id, fn, clip_id, position in entries:
                            if fn == frame_num:
                                cache_key = f"{source_id}_{frame_num}"
                                self._mem_cache[cache_key] = qimage
                                self.thumbnail_ready.emit(clip_id, position, qimage)

                # Plan and submit sweep batches
                if len(in_flight) < _MAX_WORKERS and not self._stopped:
                    self._submit_sweeps(in_flight)

                if not in_flight:
                    self._wake_event.wait(timeout=1.0)
                    self._wake_event.clear()
                else:
                    time.sleep(0.01)

        finally:
            if self._pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None

    def _submit_sweeps(self, in_flight):
        """Build sweep batches from visible frames and submit to pool."""
        with self._priority_lock:
            visible_ids = set(self._priority_clip_ids)
            playhead = self._playhead_frame

        with self._lookup_lock:
            clip_lookup = dict(self._clip_lookup)

        # Already in-flight frame keys
        in_flight_keys = set()
        for entries in in_flight.values():
            for source_id, frame_num, clip_id, position in entries:
                in_flight_keys.add((source_id, frame_num))

        # Collect all uncached visible frames, grouped by source
        by_source: Dict[str, List[Tuple[int, str, str]]] = {}
        for clip_id in visible_ids:
            clip_entries = clip_lookup.get(clip_id)
            if not clip_entries:
                continue
            for source_id, frame_num, position in clip_entries:
                if (source_id, frame_num) in in_flight_keys:
                    continue
                cache_key = f"{source_id}_{frame_num}"
                if cache_key in self._mem_cache:
                    self.thumbnail_ready.emit(
                        clip_id, position, self._mem_cache[cache_key])
                    continue
                # Emit LQ placeholder
                if (cache_key not in self._lq_emitted
                        and self._proxy_manager is not None):
                    proxy = self._proxy_manager.get_proxy(source_id)
                    if proxy:
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
                by_source.setdefault(source_id, []).append(
                    (frame_num, clip_id, position))

        if not by_source:
            return

        # LQ placeholders were emitted above; skip the expensive HQ decode
        # jobs when the user has disabled HQ generation.
        if not self._hq_enabled:
            return

        # For each source, build sweep runs ordered by playhead distance
        for source_id, frames in by_source.items():
            source = self._sources.get(source_id)
            if source is None:
                continue

            # Split into right sweep (>= playhead) and left sweep (< playhead)
            # Each sorted by frame number ascending for sequential decode
            right = sorted([f for f in frames if f[0] >= playhead], key=lambda x: x[0])
            left = sorted([f for f in frames if f[0] < playhead], key=lambda x: x[0])

            # Build runs within each sweep (group frames within SEQUENTIAL_THRESHOLD)
            for sweep_frames in [right, left]:
                runs = self._build_runs(sweep_frames)
                for run in runs:
                    if len(in_flight) >= _MAX_WORKERS:
                        return
                    target_frames = [f[0] for f in run]
                    entries = [(source_id, f[0], f[1], f[2]) for f in run]
                    # Check none are already in-flight
                    if any((source_id, fn) in in_flight_keys for fn in target_frames):
                        continue
                    for fn in target_frames:
                        in_flight_keys.add((source_id, fn))

                    if len(target_frames) == 1:
                        # Single frame — use fast independent seek
                        fut = self._pool.submit(
                            self._grab_frame_single, source, target_frames[0])
                        in_flight[fut] = entries
                    else:
                        # Multiple frames — sequential sweep decode
                        fut = self._pool.submit(
                            self._grab_frames_sweep, source, target_frames)
                        in_flight[fut] = entries

    @staticmethod
    def _build_runs(frames: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
        """Group sorted frames into runs where consecutive frames are within
        SEQUENTIAL_THRESHOLD of each other."""
        if not frames:
            return []
        runs = [[frames[0]]]
        for f in frames[1:]:
            if f[0] - runs[-1][-1][0] <= _SEQUENTIAL_THRESHOLD:
                runs[-1].append(f)
            else:
                runs.append([f])
        return runs

    def _grab_frame_single(self, source: VideoSource,
                           frame_num: int) -> Optional[List[Tuple[int, QImage]]]:
        """Grab one frame via ffmpeg seek. Used for isolated frames."""
        if self._stopped or not self._pause_event.is_set():
            return None
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
            with self._procs_lock:
                if self._stopped:
                    proc.kill()
                    return None
                self._active_procs.add(proc)
            try:
                data, _ = proc.communicate(timeout=15)
            finally:
                with self._procs_lock:
                    self._active_procs.discard(proc)
        except (OSError, subprocess.TimeoutExpired):
            if proc:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            return None

        if self._stopped or len(data) != THUMB_FRAME_SIZE:
            return None

        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            THUMB_HEIGHT, THUMB_WIDTH, 3)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()
        return [(frame_num, qimg)]

    def _grab_frames_sweep(self, source: VideoSource,
                           target_frames: List[int]) -> Optional[List[Tuple[int, QImage]]]:
        """Grab multiple frames via a single PyAV sequential decode.
        target_frames must be sorted ascending. Seeks once to the first target,
        then decodes forward through all targets."""
        if self._stopped or not self._pause_event.is_set():
            return None
        fps = source.fps if source.fps > 0 else 24.0
        results = []
        try:
            container = av.open(source.file_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)

            targets = list(target_frames)
            target_pts = [int(f / fps / tb) for f in targets]
            idx = 0  # next target to find

            # Seek to just before first target
            container.seek(target_pts[0], stream=stream)

            for frame in container.decode(stream):
                # Abort mid-sweep when the user is scrubbing — frees CPU
                # for the Qt main thread to repaint the playhead.
                if self._stopped or not self._pause_event.is_set():
                    break
                if frame.pts is None:
                    continue
                # Check if we've reached or passed the current target
                while idx < len(targets) and frame.pts >= target_pts[idx]:
                    rgb = frame.to_ndarray(format="rgb24")
                    thumb = cv2.resize(rgb, (THUMB_WIDTH, THUMB_HEIGHT),
                                       interpolation=cv2.INTER_AREA)
                    h, w, ch = thumb.shape
                    qimg = QImage(bytes(thumb.data), w, h, ch * w,
                                  QImage.Format.Format_RGB888).copy()
                    results.append((targets[idx], qimg))
                    idx += 1
                if idx >= len(targets):
                    break

            container.close()
        except Exception as e:
            logger.debug("Sweep decode error: %s", e)

        return results if results else None

    def stop(self):
        self._stopped = True
        self._pause_event.set()
        self._wake_event.set()
        # Kill any in-flight ffmpeg subprocesses so workers return immediately
        # instead of blocking on communicate().
        with self._procs_lock:
            procs = list(self._active_procs)
            self._active_procs.clear()
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        if self._coord_thread and self._coord_thread.is_alive():
            self._coord_thread.join(timeout=3.0)
        self._coord_thread = None
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def invalidate_source(self, source_id: str):
        """Remove all cached thumbnails for a source so they regenerate."""
        keys = [k for k in self._mem_cache if k.startswith(f"{source_id}_")]
        for k in keys:
            del self._mem_cache[k]
        lq_keys = [k for k in self._lq_emitted if k.startswith(f"{source_id}_")]
        for k in lq_keys:
            self._lq_emitted.discard(k)
        self._wake_event.set()

    def clear(self):
        self._mem_cache.clear()
        self._lq_emitted.clear()
