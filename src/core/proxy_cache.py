import logging
import mmap
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.video_source import VideoSource
from utils.paths import get_cache_dir

logger = logging.getLogger(__name__)

# Proxy resolution — matches TransNetV2 decode resolution
PROXY_WIDTH = 48
PROXY_HEIGHT = 27
PROXY_FRAME_SIZE = PROXY_WIDTH * PROXY_HEIGHT * 3


def _proxy_path(source: VideoSource) -> Path:
    """Path to the proxy file for a given video source."""
    import hashlib
    h = hashlib.md5(source.file_path.encode()).hexdigest()[:12]
    return get_cache_dir() / "proxies" / f"{h}.proxy"


class ProxyFile:
    """Memory-mapped flat binary file of all video frames at 48x27 RGB24.
    Provides microsecond random access to any frame — no decode needed.
    Used by TransNetV2 scene detection for fast frame reuse."""

    def __init__(self, source: VideoSource):
        self._source = source
        self._path = _proxy_path(source)
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._n_frames = 0

    @property
    def exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def open(self) -> bool:
        """Open existing proxy file for reading. Returns True if successful."""
        if not self.exists:
            return False
        try:
            file_size = self._path.stat().st_size
            self._n_frames = file_size // PROXY_FRAME_SIZE
            self._file = open(self._path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            logger.info("Opened proxy: %s (%d frames)", self._path, self._n_frames)
            return True
        except Exception as e:
            logger.warning("Failed to open proxy %s: %s", self._path, e)
            self.close()
            return False

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a frame by number. Returns RGB24 numpy array at 48x27, or None."""
        if self._mmap is None or frame_number < 0 or frame_number >= self._n_frames:
            return None
        offset = frame_number * PROXY_FRAME_SIZE
        raw = self._mmap[offset:offset + PROXY_FRAME_SIZE]
        return np.frombuffer(raw, dtype=np.uint8).reshape(PROXY_HEIGHT, PROXY_WIDTH, 3).copy()

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()

    @staticmethod
    def save_frames(source: VideoSource, frames: list):
        """Save a list of RGB24 numpy arrays (48x27) as a proxy file."""
        path = _proxy_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                for frame in frames:
                    f.write(frame.tobytes())
            logger.info("Saved proxy: %s (%d frames, %.0f MB)",
                         path, len(frames), path.stat().st_size / 1e6)
        except OSError:
            # File may be mmap'd by ProxyManager — skip, existing proxy still works
            logger.warning("Proxy file locked (mmap'd), skipping save for %s", path)


class ProxyManager:
    """Manages .proxy files for scene detection reuse."""

    def __init__(self):
        self._proxies: Dict[str, ProxyFile] = {}

    def get_proxy(self, source_id: str) -> Optional[ProxyFile]:
        return self._proxies.get(source_id)

    def load_or_open(self, source: VideoSource) -> Optional[ProxyFile]:
        """Open .proxy file for a source. Returns None if no proxy exists."""
        if source.id in self._proxies:
            return self._proxies[source.id]
        proxy = ProxyFile(source)
        if proxy.open():
            self._proxies[source.id] = proxy
            return proxy
        return None

    def register(self, source: VideoSource, proxy: ProxyFile):
        self._proxies[source.id] = proxy

    def close_all(self):
        for proxy in self._proxies.values():
            proxy.close()
        self._proxies.clear()
