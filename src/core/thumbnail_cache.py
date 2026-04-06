import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from core.timeline import TimelineModel
from core.video_source import VideoSource

logger = logging.getLogger(__name__)

THUMB_WIDTH = 96
THUMB_HEIGHT = 54


class ThumbnailCache(QObject):
    """Background thumbnail generation using per-source FFmpeg select filter.

    For each source, spawns one FFmpeg process that seeks only the needed frames
    using select='eq(n,F1)+eq(n,F2)+...' and outputs MJPEG via pipe.
    Low-priority background — yields to scrubbing and playback.
    No disk cache — thumbnails regenerated fresh each session.
    """

    thumbnail_ready = Signal(str, str, QImage)  # clip_id, "first"|"last", qimage

    def __init__(self, timeline: TimelineModel,
                 sources: Dict[str, VideoSource], parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._mem_cache: Dict[str, QImage] = {}
        self._stopped = False
        self._coord_thread: Optional[threading.Thread] = None
        self._active_proc: Optional[subprocess.Popen] = None

        # Pause support
        self._pause_event = threading.Event()
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def generate_all(self):
        """Start thumbnail generation in a background coordinator thread."""
        if self._coord_thread and self._coord_thread.is_alive():
            return
        self._stopped = False
        self._coord_thread = threading.Thread(target=self._coordinate, daemon=True)
        self._coord_thread.start()

    def _coordinate(self):
        """Collect needed frames per source, generate via select-filter MJPEG."""
        # Group needed frames by source
        source_requests: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for clip in self._timeline.clips:
            if clip.is_gap or self._stopped:
                continue
            source_requests[clip.source_id].append(
                (clip.source_in, clip.id, "first"))
            source_requests[clip.source_id].append(
                (clip.source_out, clip.id, "last"))

        if not source_requests or self._stopped:
            return

        # Emit any already-cached thumbnails immediately
        remaining: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for source_id, requests in source_requests.items():
            for frame_num, clip_id, position in requests:
                key = f"{source_id}_{frame_num}"
                if key in self._mem_cache:
                    self.thumbnail_ready.emit(
                        clip_id, position, self._mem_cache[key])
                else:
                    remaining[source_id].append((frame_num, clip_id, position))

        if not remaining:
            return

        # Process each source with one FFmpeg select-filter process
        for source_id, requests in remaining.items():
            if self._stopped:
                return
            source = self._sources.get(source_id)
            if source is None:
                continue
            self._generate_for_source(source, requests)

    def _generate_for_source(self, source: VideoSource,
                             requests: List[Tuple[int, str, str]]):
        """One FFmpeg process per source: select only needed frames as MJPEG."""
        # Deduplicate frame numbers and sort (output arrives in source order)
        frame_to_requests: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        for frame_num, clip_id, position in requests:
            frame_to_requests[frame_num].append((clip_id, position))
        unique_frames = sorted(frame_to_requests.keys())

        if not unique_frames or self._stopped:
            return

        # Build select expression and write to filter script file
        # (avoids Windows command-line length limits for large timelines)
        select_terms = "+".join(f"eq(n\\,{f})" for f in unique_frames)
        vf = (f"select='{select_terms}',"
              f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear")

        filter_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="prismasynth_vf_",
            delete=False, encoding="utf-8")
        filter_file.write(vf)
        filter_file.close()
        filter_path = filter_file.name

        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-hwaccel", "cuda",
            "-i", source.file_path,
            "-filter_script:v", filter_path, "-vsync", "drop",
            "-c:v", "mjpeg", "-q:v", "5",
            "-f", "image2pipe", "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            logger.warning("Failed to start FFmpeg for thumbnails: %s", e)
            try:
                os.unlink(filter_path)
            except OSError:
                pass
            return
        self._active_proc = proc

        # Check if FFmpeg exits immediately (bad filter syntax)
        time.sleep(0.2)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")[:500]
            logger.warning("FFmpeg thumbnail process exited immediately: %s",
                           stderr)
            try:
                os.unlink(filter_path)
            except OSError:
                pass
            return

        # Parse MJPEG stream — frames arrive in ascending source frame order
        buf = bytearray()
        frame_idx = 0
        SOI = b'\xff\xd8'
        EOI = b'\xff\xd9'

        try:
            while not self._stopped and frame_idx < len(unique_frames):
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                buf.extend(chunk)

                # Extract complete JPEG frames
                while frame_idx < len(unique_frames):
                    soi = buf.find(SOI)
                    if soi < 0:
                        break
                    eoi = buf.find(EOI, soi + 2)
                    if eoi < 0:
                        break

                    jpeg_data = bytes(buf[soi:eoi + 2])
                    del buf[:eoi + 2]

                    # Map this JPEG to the corresponding source frame number
                    frame_num = unique_frames[frame_idx]
                    frame_idx += 1

                    qimage = self._jpeg_to_qimage(jpeg_data)
                    if qimage is None:
                        continue

                    # Cache and emit for all clips that need this frame
                    key = f"{source.id}_{frame_num}"
                    self._mem_cache[key] = qimage
                    for clip_id, position in frame_to_requests[frame_num]:
                        if self._stopped:
                            return
                        # Respect pause (scrubbing/playback priority)
                        while not self._pause_event.wait(timeout=0.1):
                            if self._stopped:
                                return
                        self.thumbnail_ready.emit(clip_id, position, qimage)

                    # Brief yield so main thread stays responsive
                    time.sleep(0.002)

        except Exception as e:
            logger.debug("Thumbnail generation error: %s", e)
        finally:
            self._active_proc = None
            try:
                proc.stdout.close()
                proc.kill()
                proc.wait()
            except Exception:
                pass
            try:
                os.unlink(filter_path)
            except OSError:
                pass

        logger.info("Generated %d thumbnails for %s",
                    frame_idx, source.file_path)

    @staticmethod
    def _jpeg_to_qimage(jpeg_data: bytes) -> Optional[QImage]:
        """Decode JPEG bytes to QImage."""
        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        bgr = __import__('cv2').imdecode(arr, __import__('cv2').IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = __import__('cv2').cvtColor(bgr, __import__('cv2').COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # .copy() so QImage owns its pixel data (prevents use-after-free)
        return QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()

    def stop(self):
        self._stopped = True
        self._pause_event.set()
        if self._active_proc:
            try:
                self._active_proc.kill()
            except OSError:
                pass
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
