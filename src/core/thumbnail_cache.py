import hashlib
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from core.clip import Clip
from core.proxy_cache import ProxyManager
from core.timeline import TimelineModel
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool
from utils.paths import get_thumbnail_dir

logger = logging.getLogger(__name__)

THUMB_WIDTH = 96
THUMB_HEIGHT = 54
THUMB_FRAME_SIZE = THUMB_WIDTH * THUMB_HEIGHT * 3


def _source_hash(source: VideoSource) -> str:
    """Short hash of source path for disk cache directory."""
    return hashlib.md5(source.file_path.encode()).hexdigest()[:12]


class ThumbnailCache(QObject):
    """Fast thumbnail generation using parallel ffmpeg input-seeking.

    Each thumbnail is grabbed via a separate ffmpeg -ss seek (0.2-0.3s each).
    4 workers run in parallel for ~25s total on a 2-hour 4K movie (300 thumbnails).
    Results are cached to disk (JPEG) for instant reload on subsequent sessions.
    Supports pause/resume to yield CPU to interactive scrubbing.
    """

    thumbnail_ready = Signal(str, str, QImage)  # clip_id, "first"|"last", qimage

    def __init__(self, reader_pool: VideoReaderPool, timeline: TimelineModel,
                 sources: Dict[str, VideoSource],
                 proxy_manager: Optional[ProxyManager] = None, parent=None):
        super().__init__(parent)
        self._reader_pool = reader_pool
        self._timeline = timeline
        self._sources = sources
        self._proxy_manager = proxy_manager
        self._mem_cache: Dict[str, QImage] = {}
        self._stopped = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._coord_thread: Optional[threading.Thread] = None

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
        """Coordinator: collect needed thumbnails, check disk cache, dispatch seeks."""
        # Collect all needed thumbnails: (source_id, frame_num, clip_id, position)
        requests: List[Tuple[str, int, str, str]] = []
        for clip in self._timeline.clips:
            if clip.is_gap or self._stopped:
                continue
            requests.append((clip.source_id, clip.source_in, clip.id, "first"))
            requests.append((clip.source_id, clip.source_out, clip.id, "last"))

        if not requests or self._stopped:
            return

        # Check disk cache and proxy first — emit cached thumbnails immediately
        remaining = []
        for source_id, frame_num, clip_id, position in requests:
            if self._stopped:
                return
            source = self._sources.get(source_id)
            if source is None:
                continue
            disk_dir = get_thumbnail_dir() / _source_hash(source)
            qimage = self._load_from_disk(disk_dir, frame_num)
            if qimage is None and self._proxy_manager:
                # Try extracting thumbnail from JPEG proxy (sub-millisecond)
                proxy = self._proxy_manager.get_proxy(source_id)
                if proxy is not None and frame_num < proxy.n_frames:
                    frame_rgb = proxy.get_frame(frame_num)
                    if frame_rgb is not None:
                        thumb = cv2.resize(
                            frame_rgb, (THUMB_WIDTH, THUMB_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                        disk_dir.mkdir(parents=True, exist_ok=True)
                        self._save_to_disk(disk_dir, frame_num, thumb)
                        qimage = self._ndarray_to_qimage(thumb)
            if qimage is not None:
                self._mem_cache[f"{source_id}_{frame_num}"] = qimage
                self.thumbnail_ready.emit(clip_id, position, qimage)
            else:
                remaining.append((source_id, frame_num, clip_id, position, source))

        if not remaining or self._stopped:
            logger.info("All %d thumbnails loaded from disk cache", len(requests))
            return

        logger.info("Loaded %d thumbnails from cache, generating %d remaining",
                     len(requests) - len(remaining), len(remaining))

        # Generate remaining thumbnails with parallel ffmpeg seeks
        self._executor = ThreadPoolExecutor(max_workers=4)
        try:
            futures: List[Tuple[Future, str, str, str, int, Path]] = []
            for source_id, frame_num, clip_id, position, source in remaining:
                if self._stopped:
                    return
                disk_dir = get_thumbnail_dir() / _source_hash(source)
                disk_dir.mkdir(parents=True, exist_ok=True)
                fut = self._executor.submit(
                    self._grab_frame, source.file_path, source.fps,
                    frame_num, disk_dir
                )
                futures.append((fut, clip_id, position, source_id, frame_num, disk_dir))

            # Collect results as they complete — yield between emissions
            for fut, clip_id, position, source_id, frame_num, disk_dir in futures:
                if self._stopped:
                    return
                # Wait if paused (scrubbing in progress)
                while not self._pause_event.wait(timeout=0.1):
                    if self._stopped:
                        return
                try:
                    qimage = fut.result(timeout=10)
                    if qimage is not None and not self._stopped:
                        # Re-check pause — may have been paused while waiting on result
                        while not self._pause_event.wait(timeout=0.1):
                            if self._stopped:
                                return
                        self._mem_cache[f"{source_id}_{frame_num}"] = qimage
                        self.thumbnail_ready.emit(clip_id, position, qimage)
                        # Brief yield so main thread can process scrub signals first
                        time.sleep(0.005)
                except Exception as e:
                    logger.debug("Thumbnail failed for frame %d: %s", frame_num, e)

        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

        logger.info("Generated %d thumbnails", len(remaining))

    def _grab_frame(self, file_path: str, fps: float, frame_num: int,
                    disk_dir: Path) -> Optional[QImage]:
        """Grab a single frame at thumbnail resolution via ffmpeg -ss input seeking."""
        if self._stopped:
            return None

        timestamp = frame_num / fps if fps > 0 else 0.0
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-ss", f"{timestamp:.4f}",
            "-i", file_path,
            "-frames:v", "1",
            "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = proc.stdout.read()
        proc.wait()

        if len(data) != THUMB_FRAME_SIZE:
            return None

        frame_rgb = np.frombuffer(data, dtype=np.uint8).reshape(THUMB_HEIGHT, THUMB_WIDTH, 3)

        # Save to disk cache
        self._save_to_disk(disk_dir, frame_num, frame_rgb)

        return self._ndarray_to_qimage(frame_rgb)

    # --- Disk cache ---

    def _disk_path(self, disk_dir: Path, frame_number: int) -> Path:
        return disk_dir / f"{frame_number}.jpg"

    def _save_to_disk(self, disk_dir: Path, frame_number: int, frame_rgb: np.ndarray):
        path = self._disk_path(disk_dir, frame_number)
        try:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            logger.debug("Disk cache write failed: %s", e)

    def _load_from_disk(self, disk_dir: Path, frame_number: int) -> Optional[QImage]:
        path = self._disk_path(disk_dir, frame_number)
        if not path.exists():
            return None
        try:
            bgr = cv2.imread(str(path))
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return self._ndarray_to_qimage(rgb)
        except Exception:
            return None

    # --- Utils ---

    @staticmethod
    def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        return QImage(bytes(arr.data), w, h, bytes_per_line, QImage.Format.Format_RGB888)

    def stop(self):
        self._stopped = True
        self._pause_event.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self._coord_thread and self._coord_thread.is_alive():
            self._coord_thread.join(timeout=2.0)
        self._coord_thread = None

    def clear(self):
        self._mem_cache.clear()
