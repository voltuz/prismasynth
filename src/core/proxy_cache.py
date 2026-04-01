import logging
import mmap
import shutil
import struct
import threading
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.video_source import VideoSource
from utils.paths import get_cache_dir

logger = logging.getLogger(__name__)

# Proxy resolution — matches TransNetV2 decode resolution
PROXY_WIDTH = 48
PROXY_HEIGHT = 27
PROXY_FRAME_SIZE = PROXY_WIDTH * PROXY_HEIGHT * 3

# JPEG proxy — high-quality scrub proxy (quarter-HD)
JPROXY_MAGIC = b"JPXY"
JPROXY_VERSION = 1
JPROXY_WIDTH = 960
JPROXY_HEIGHT = 540
JPROXY_QUALITY = 80
JPROXY_HEADER_SIZE = 18  # magic(4) + version(2) + width(2) + height(2) + reserved(4) + n_frames(4)


def _proxy_path(source: VideoSource) -> Path:
    """Path to the proxy file for a given video source."""
    import hashlib
    h = hashlib.md5(source.file_path.encode()).hexdigest()[:12]
    return get_cache_dir() / "proxies" / f"{h}.proxy"


class ProxyFile:
    """Memory-mapped flat binary file of all video frames at 48x27 RGB24.
    Provides microsecond random access to any frame — no decode needed."""

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


def _jproxy_path(source: VideoSource) -> Path:
    """Path to the JPEG proxy file — stored next to the original video."""
    video = Path(source.file_path)
    return video.with_suffix(".jproxy")


class JpegProxyFile:
    """Memory-mapped JPEG-indexed proxy file for high-quality scrub preview.

    Format: fixed header + (N+1) uint64 offset table + concatenated JPEG blobs.
    Provides sub-millisecond random access to any frame at 960x540."""

    def __init__(self, source: VideoSource):
        self._source = source
        self._path = _jproxy_path(source)
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._n_frames = 0
        self._width = 0
        self._height = 0

    @property
    def exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > JPROXY_HEADER_SIZE

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def open(self) -> bool:
        """Open existing JPEG proxy file for reading. Returns True if successful."""
        if not self.exists:
            return False
        try:
            self._file = open(self._path, "rb")
            header = self._file.read(JPROXY_HEADER_SIZE)
            if len(header) < JPROXY_HEADER_SIZE:
                self.close()
                return False

            magic = header[0:4]
            if magic != JPROXY_MAGIC:
                logger.warning("Bad magic in jproxy %s", self._path)
                self.close()
                return False

            version = struct.unpack_from("<H", header, 4)[0]
            if version != JPROXY_VERSION:
                logger.warning("Unsupported jproxy version %d in %s", version, self._path)
                self.close()
                return False

            self._width = struct.unpack_from("<H", header, 6)[0]
            self._height = struct.unpack_from("<H", header, 8)[0]
            # bytes 10-13: reserved
            self._n_frames = struct.unpack_from("<I", header, 14)[0]

            if self._n_frames == 0:
                self.close()
                return False

            # mmap the whole file for fast offset+JPEG reads
            self._file.seek(0)
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            logger.info("Opened jproxy: %s (%d frames, %dx%d)",
                        self._path, self._n_frames, self._width, self._height)
            return True
        except Exception as e:
            logger.warning("Failed to open jproxy %s: %s", self._path, e)
            self.close()
            return False

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a frame by number. Returns RGB numpy array, or None."""
        if self._mmap is None or frame_number < 0 or frame_number >= self._n_frames:
            return None
        try:
            # Read offset[frame_number] and offset[frame_number + 1] from the offset table
            off_pos = JPROXY_HEADER_SIZE + frame_number * 8
            start = struct.unpack_from("<Q", self._mmap, off_pos)[0]
            end = struct.unpack_from("<Q", self._mmap, off_pos + 8)[0]

            jpeg_bytes = self._mmap[start:end]
            bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

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


class JpegProxyWriter:
    """Streaming writer for JPEG proxy files.

    Writes JPEG-compressed frames to a temp file during scene detection,
    then assembles the final .jproxy with header + offset table + data."""

    def __init__(self, path: Path, width: int, height: int):
        self._path = path
        self._width = width
        self._height = height
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = path.with_suffix(".jproxy.tmp")
        self._temp_file = open(self._temp_path, "wb")
        self._offsets: list = []  # byte offset of each JPEG blob in temp file
        self._current_offset = 0

    def write_frame(self, frame_rgb: np.ndarray):
        """JPEG-encode an RGB frame and append to the temp file."""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        success, jpeg_buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPROXY_QUALITY]
        )
        if not success:
            raise RuntimeError(f"JPEG encode failed for frame {len(self._offsets)}")
        data = jpeg_buf.tobytes()
        self._offsets.append(self._current_offset)
        self._temp_file.write(data)
        self._current_offset += len(data)

    def finish(self):
        """Assemble the final .jproxy file: header + offset table + JPEG data."""
        if self._temp_file is not None:
            self._temp_file.close()
            self._temp_file = None
        n_frames = len(self._offsets)
        if n_frames == 0:
            self.abort()
            return

        # Compute the data start position in the final file
        offset_table_size = (n_frames + 1) * 8  # N+1 entries (sentinel)
        data_start = JPROXY_HEADER_SIZE + offset_table_size

        # Adjust offsets: temp file offsets are relative to 0, final offsets relative to file start
        final_offsets = [off + data_start for off in self._offsets]
        final_offsets.append(self._current_offset + data_start)  # sentinel

        # Write final file
        with open(self._path, "wb") as f:
            # Header
            f.write(JPROXY_MAGIC)
            f.write(struct.pack("<H", JPROXY_VERSION))
            f.write(struct.pack("<H", self._width))
            f.write(struct.pack("<H", self._height))
            f.write(struct.pack("<I", 0))  # reserved
            f.write(struct.pack("<I", n_frames))

            # Offset table (single write for efficiency)
            import array
            offset_arr = array.array('Q', final_offsets)
            f.write(offset_arr.tobytes())

            # Copy JPEG data from temp file
            with open(self._temp_path, "rb") as tmp:
                shutil.copyfileobj(tmp, f, length=1024 * 1024)

        file_size = self._path.stat().st_size
        logger.info("Saved jproxy: %s (%d frames, %dx%d, %.0f MB)",
                     self._path, n_frames, self._width, self._height,
                     file_size / 1e6)

        # Clean up temp file
        try:
            self._temp_path.unlink()
        except Exception:
            pass

    def abort(self):
        """Clean up on cancellation or error."""
        if self._temp_file is not None:
            try:
                self._temp_file.close()
            except Exception:
                pass
            self._temp_file = None
        for p in [self._temp_path, self._path]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


class HQProxyGenerator(QObject):
    """Background generator for high-quality JPEG proxy files.

    Runs after scene detection completes so it doesn't slow down cut detection.
    Decodes video at 960x540 via ffmpeg and streams JPEG frames to a .jproxy file."""

    finished = Signal(str)  # source_id when done
    progress = Signal(str, int)  # source_id, percent

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threads: Dict[str, threading.Thread] = {}
        self._cancelled = False
        self._procs: list = []

    def generate(self, source: VideoSource):
        """Start background HQ proxy generation for a source."""
        self._cancelled = False
        jpath = _jproxy_path(source)
        if jpath.exists() and jpath.stat().st_size > JPROXY_HEADER_SIZE:
            logger.info("HQ proxy already exists for %s", source.file_path)
            self.finished.emit(source.id)
            return
        if source.id in self._threads and self._threads[source.id].is_alive():
            return
        t = threading.Thread(
            target=self._generate_worker, args=(source,), daemon=True
        )
        self._threads[source.id] = t
        t.start()

    def _generate_worker(self, source: VideoSource):
        """Worker thread: decode at 960x540 and write .jproxy."""
        import subprocess
        proxy_frame_size = JPROXY_WIDTH * JPROXY_HEIGHT * 3
        total = source.total_frames

        # Try GPU decode, fall back to CPU
        cmd = [
            "ffmpeg", "-v", "quiet", "-hwaccel", "cuda",
            "-i", source.file_path,
            "-vf", f"scale={JPROXY_WIDTH}:{JPROXY_HEIGHT}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._procs.append(proc)

        test_data = proc.stdout.read(proxy_frame_size * 2)
        if len(test_data) < proxy_frame_size and proc.poll() is not None:
            logger.info("HQ proxy: GPU unavailable, using CPU for %s", source.file_path)
            proc.stdout.close()
            proc.wait()
            cmd = [
                "ffmpeg", "-v", "quiet", "-threads", "0",
                "-i", source.file_path,
                "-vf", f"scale={JPROXY_WIDTH}:{JPROXY_HEIGHT}:flags=fast_bilinear",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._procs.append(proc)
            test_data = b""

        writer = JpegProxyWriter(
            _jproxy_path(source), JPROXY_WIDTH, JPROXY_HEIGHT
        )

        buf = bytearray(test_data)
        chunk_bytes = proxy_frame_size * 10
        n_frames = 0

        try:
            while not self._cancelled:
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= proxy_frame_size:
                    frame_rgb = (
                        np.frombuffer(bytes(buf[:proxy_frame_size]), dtype=np.uint8)
                        .reshape(JPROXY_HEIGHT, JPROXY_WIDTH, 3)
                    )
                    del buf[:proxy_frame_size]
                    writer.write_frame(frame_rgb)
                    n_frames += 1
                    if total > 0 and n_frames % max(1, total // 100) == 0:
                        self.progress.emit(source.id, min(99, int(n_frames / total * 100)))

            proc.stdout.close()
            proc.wait()

            if self._cancelled:
                writer.abort()
                return

            writer.finish()
            logger.info("HQ proxy generated for %s (%d frames)", source.file_path, n_frames)
            self.finished.emit(source.id)

        except Exception as e:
            logger.exception("HQ proxy generation failed for %s", source.file_path)
            writer.abort()

    def cancel(self):
        self._cancelled = True
        for proc in self._procs:
            try:
                proc.kill()
            except Exception:
                pass

    def stop(self):
        self.cancel()
        for t in self._threads.values():
            t.join(timeout=3.0)
        self._threads.clear()
        self._procs.clear()


class ProxyManager:
    """Manages proxy files for multiple video sources."""

    def __init__(self):
        self._proxies: Dict[str, Union[ProxyFile, JpegProxyFile]] = {}

    def get_proxy(self, source_id: str) -> Optional[Union[ProxyFile, JpegProxyFile]]:
        return self._proxies.get(source_id)

    def load_or_open(self, source: VideoSource) -> Optional[Union[ProxyFile, JpegProxyFile]]:
        """Open proxy file for a source. Prefers .jproxy over .proxy.
        Returns None if no proxy exists."""
        if source.id in self._proxies:
            return self._proxies[source.id]
        # Try JPEG proxy first (higher quality)
        jproxy = JpegProxyFile(source)
        if jproxy.open():
            self._proxies[source.id] = jproxy
            return jproxy
        # Fall back to legacy raw proxy
        proxy = ProxyFile(source)
        if proxy.open():
            self._proxies[source.id] = proxy
            return proxy
        return None

    def upgrade_to_hq(self, source: VideoSource) -> bool:
        """Replace legacy proxy with HQ JPEG proxy if available. Returns True if upgraded."""
        jproxy = JpegProxyFile(source)
        if jproxy.open():
            old = self._proxies.get(source.id)
            if old is not None:
                old.close()
            self._proxies[source.id] = jproxy
            logger.info("Upgraded proxy to HQ for %s", source.id)
            return True
        return False

    def register(self, source: VideoSource, proxy: Union[ProxyFile, JpegProxyFile]):
        self._proxies[source.id] = proxy

    def close_all(self):
        for proxy in self._proxies.values():
            proxy.close()
        self._proxies.clear()
