import logging
import threading
from collections import OrderedDict
from fractions import Fraction
from typing import Iterator, List, Optional, Dict, Tuple

import av
import numpy as np

from core.video_source import VideoSource

logger = logging.getLogger(__name__)

FRAME_CACHE_SIZE = 500  # number of decoded frames to keep in memory


class FrameCache:
    """Thread-safe LRU cache for decoded video frames."""

    def __init__(self, maxsize: int = FRAME_CACHE_SIZE):
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, source_id: str, frame_number: int) -> Optional[np.ndarray]:
        key = (source_id, frame_number)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def get_nearest(self, source_id: str, frame_number: int, tolerance: int = 3) -> Optional[Tuple[int, np.ndarray]]:
        """Get cached frame closest to frame_number within tolerance."""
        with self._lock:
            best_key = None
            best_dist = tolerance + 1
            for key in self._cache:
                if key[0] == source_id:
                    dist = abs(key[1] - frame_number)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = key
            if best_key is not None:
                self._cache.move_to_end(best_key)
                return (best_key[1], self._cache[best_key])
        return None

    def put(self, source_id: str, frame_number: int, frame: np.ndarray):
        key = (source_id, frame_number)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            self._cache[key] = frame
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()


# Shared global frame cache
_frame_cache = FrameCache()


def get_frame_cache() -> FrameCache:
    return _frame_cache


class VideoReader:
    """Decodes video frames using PyAV with multi-threaded CPU decoding."""

    def __init__(self, source: VideoSource, use_gpu: bool = True):
        self._source = source
        self._lock = threading.Lock()
        self._container: Optional[av.container.InputContainer] = None
        self._stream = None
        self._use_gpu = use_gpu
        self._last_decoded_frame_num = -1
        self._open()

    def _open(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass

        self._container = av.open(self._source.file_path)
        self._stream = self._container.streams.video[0]
        # Enable multi-threaded decoding for speed
        self._stream.thread_type = "AUTO"
        self._stream.thread_count = 0  # auto-detect thread count
        self._last_decoded_frame_num = -1

        # Precompute precise time_base as Fraction to avoid float drift
        tb = self._stream.time_base
        self._tb_frac = Fraction(tb.numerator, tb.denominator) if tb else Fraction(1, 24000)
        self._fps_frac = Fraction(self._source.fps).limit_denominator(100000)

        logger.info("Opened %s (threads=auto, tb=%s, fps=%s)",
                     self._source.file_path, self._tb_frac, self._fps_frac)

    def _frame_number_from_pts(self, frame) -> int:
        """Convert a frame's PTS to a frame number using exact rational arithmetic."""
        if frame.pts is not None:
            # pts * time_base * fps = frame number (exact with Fraction)
            return int(Fraction(frame.pts) * self._tb_frac * self._fps_frac + Fraction(1, 10))
        return -1

    def _seek_to(self, frame_number: int):
        """Seek to the nearest keyframe at or before frame_number."""
        if self._fps_frac > 0:
            # frame_number / fps / time_base = target_pts (exact)
            target_pts = int(Fraction(frame_number) / self._fps_frac / self._tb_frac)
            self._container.seek(target_pts, stream=self._stream)
        else:
            self._container.seek(0)

    def seek_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Seek to a specific frame and return as RGB numpy array. Uses cache."""
        frame_number = max(0, min(frame_number, self._source.total_frames - 1))

        # Check cache first
        cached = _frame_cache.get(self._source.id, frame_number)
        if cached is not None:
            return cached

        with self._lock:
            return self._seek_frame_locked(frame_number)

    def _seek_frame_locked(self, frame_number: int) -> Optional[np.ndarray]:
        """Internal seek — must be called with lock held."""
        try:
            # If we're close to where we left off and going forward, just decode forward
            # instead of seeking (avoids keyframe decode overhead). At 64 frames this
            # covers typical GOP sizes and caches all intermediate frames along the way.
            gap = frame_number - self._last_decoded_frame_num
            if 0 < gap <= 64:
                return self._decode_forward_to(frame_number)

            # Full seek needed
            self._seek_to(frame_number)
            return self._decode_forward_to(frame_number)
        except Exception as e:
            logger.warning("Seek to frame %d failed: %s, reopening", frame_number, e)
            try:
                self._open()
                self._seek_to(frame_number)
                return self._decode_forward_to(frame_number)
            except Exception as e2:
                logger.error("Seek retry failed: %s", e2)
        return None

    def _decode_forward_to(self, target_frame: int) -> Optional[np.ndarray]:
        """Decode frames until we reach or pass target_frame. Cache ALL along the way.
        Every seek decodes through the GOP — caching intermediates gives free
        backward-scrub coverage (~20-48 frames per seek instead of 2)."""
        result = None
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                current = self._frame_number_from_pts(frame)
                if current < 0:
                    current = target_frame

                arr = frame.to_ndarray(format="rgb24")
                _frame_cache.put(self._source.id, current, arr)
                self._last_decoded_frame_num = current

                if current >= target_frame:
                    return arr
                result = arr
        return result

    def read_sequential_into_buffer(self, start_frame: int, buffer: list,
                                     max_frames: int, stop_event: threading.Event,
                                     buffer_lock: threading.Lock):
        """Decode frames sequentially into a shared buffer. For playback engine.
        Stops when buffer is full, stop_event is set, or max_frames reached."""
        with self._lock:
            try:
                self._seek_to(start_frame)
                started = False
                frames_read = 0

                for packet in self._container.demux(self._stream):
                    if stop_event.is_set():
                        return
                    for frame in packet.decode():
                        if stop_event.is_set():
                            return

                        current = self._frame_number_from_pts(frame)
                        if current < 0:
                            current = start_frame
                            started = True

                        if not started:
                            if current >= start_frame:
                                started = True
                            else:
                                continue

                        arr = frame.to_ndarray(format="rgb24")
                        _frame_cache.put(self._source.id, current, arr)

                        # Wait if buffer is full
                        while not stop_event.is_set():
                            with buffer_lock:
                                if len(buffer) < max_frames:
                                    buffer.append((current, arr))
                                    break
                            stop_event.wait(0.005)

                        frames_read += 1
                        self._last_decoded_frame_num = current

                        if frames_read >= max_frames:
                            return
            except Exception as e:
                if not stop_event.is_set():
                    logger.error("Sequential read from frame %d failed: %s", start_frame, e)

    def read_sequential(self, start_frame: int, count: int) -> List[np.ndarray]:
        """Read frames sequentially. Returns a list (not generator) so the lock is released."""
        frames_out = []
        with self._lock:
            try:
                self._seek_to(start_frame)
                started = False
                for packet in self._container.demux(self._stream):
                    for frame in packet.decode():
                        current = self._frame_number_from_pts(frame)
                        if current < 0:
                            current = start_frame
                            started = True
                        if not started and current >= start_frame:
                            started = True
                        if started:
                            frames_out.append(frame.to_ndarray(format="rgb24"))
                            if len(frames_out) >= count:
                                return frames_out
            except Exception as e:
                logger.error("Sequential read from frame %d failed: %s", start_frame, e)
        return frames_out

    def get_thumbnail(self, frame_number: int, size: tuple) -> Optional[np.ndarray]:
        """Get a resized thumbnail for the given frame."""
        import cv2
        frame = self.seek_frame(frame_number)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        target_w, target_h = size
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        with self._lock:
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
                self._container = None
                self._stream = None



class VideoReaderPool:
    """Manages VideoReaders, one per VideoSource. Thread-safe.
    Creates separate reader instances for playback vs scrubbing to avoid lock contention."""

    def __init__(self, use_gpu: bool = True):
        self._readers: Dict[str, VideoReader] = {}
        self._playback_readers: Dict[str, VideoReader] = {}
        self._sources: Dict[str, VideoSource] = {}
        self._use_gpu = use_gpu
        self._lock = threading.Lock()

    def register_source(self, source: VideoSource):
        with self._lock:
            self._sources[source.id] = source

    def get_reader(self, source_id: str) -> Optional[VideoReader]:
        """Get reader for scrubbing/thumbnails."""
        with self._lock:
            if source_id in self._readers:
                return self._readers[source_id]
            source = self._sources.get(source_id)
            if source is None:
                return None
            reader = VideoReader(source, self._use_gpu)
            self._readers[source_id] = reader
            return reader

    def get_playback_reader(self, source_id: str) -> Optional[VideoReader]:
        """Get a separate reader for playback (avoids lock contention with scrub reader)."""
        with self._lock:
            if source_id in self._playback_readers:
                return self._playback_readers[source_id]
            source = self._sources.get(source_id)
            if source is None:
                return None
            reader = VideoReader(source, self._use_gpu)
            self._playback_readers[source_id] = reader
            return reader

    def get_source(self, source_id: str) -> Optional[VideoSource]:
        with self._lock:
            return self._sources.get(source_id)

    def close_all(self):
        with self._lock:
            for reader in self._readers.values():
                reader.close()
            for reader in self._playback_readers.values():
                reader.close()
            self._readers.clear()
            self._playback_readers.clear()

    def remove_source(self, source_id: str):
        with self._lock:
            for pool in (self._readers, self._playback_readers):
                reader = pool.pop(source_id, None)
                if reader:
                    reader.close()
            self._sources.pop(source_id, None)
