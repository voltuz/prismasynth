import logging
import threading
import time
from typing import Optional, Dict

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from core.timeline import TimelineModel
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool, get_frame_cache
from core.proxy_cache import ProxyManager

logger = logging.getLogger(__name__)

PREFETCH_BUFFER_SIZE = 60  # frames to buffer ahead


class PlaybackEngine(QObject):
    """Handles smooth video playback with a background prefetch thread.

    The prefetch thread decodes frames sequentially into a ring buffer.
    A QTimer on the main thread pops frames from the buffer at the correct FPS.
    This decouples decode speed from display rate.
    """

    frame_ready = Signal(int, np.ndarray)  # (timeline_frame, frame_data)
    playback_finished = Signal()

    def __init__(self, timeline: TimelineModel, sources: Dict[str, VideoSource],
                 reader_pool: VideoReaderPool, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._reader_pool = reader_pool

        self._playing = False
        self._fps = 24.0
        self._current_frame = 0

        # Prefetch buffer: list of (timeline_frame, ndarray)
        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Display timer
        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._on_display_tick)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, start_frame: int, fps: float):
        """Start playback from the given timeline frame."""
        if self._playing:
            self.stop()

        self._fps = fps if fps > 0 else 24.0
        self._current_frame = start_frame
        self._playing = True

        # Clear buffer and start prefetch
        with self._buffer_lock:
            self._buffer.clear()
        self._stop_event.clear()

        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(start_frame,),
            daemon=True,
        )
        self._prefetch_thread.start()

        # Start display timer
        interval_ms = max(1, int(1000.0 / self._fps))
        self._display_timer.start(interval_ms)

    def stop(self):
        """Stop playback."""
        self._playing = False
        self._display_timer.stop()
        self._stop_event.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None
        with self._buffer_lock:
            self._buffer.clear()

    def _on_display_tick(self):
        """Called by QTimer — pull next frame from buffer and emit."""
        frame_data = None
        timeline_frame = self._current_frame

        with self._buffer_lock:
            if self._buffer:
                timeline_frame, frame_data = self._buffer.pop(0)

        if frame_data is not None:
            self._current_frame = timeline_frame + 1
            self.frame_ready.emit(timeline_frame, frame_data)
        else:
            # Buffer empty — check if we're done
            total = self._timeline.get_total_duration_frames()
            if self._current_frame >= total:
                self.stop()
                self.playback_finished.emit()
            # else: buffer underrun — skip this tick, prefetch will catch up

    def _prefetch_worker(self, start_frame: int):
        """Background thread: decode frames sequentially into buffer."""
        timeline_frame = start_frame
        total = self._timeline.get_total_duration_frames()
        cache = get_frame_cache()

        while timeline_frame < total and not self._stop_event.is_set():
            # Map timeline frame to source
            result = self._timeline.timeline_frame_to_source_frame(timeline_frame)
            if result is None:
                timeline_frame += 1
                continue

            clip, source_frame = result

            # How many frames left in this clip?
            clip_start = self._timeline.get_clip_timeline_start(clip.id)
            offset_in_clip = timeline_frame - clip_start
            frames_left_in_clip = clip.duration_frames - offset_in_clip

            # Decode this clip segment sequentially
            reader = self._reader_pool.get_playback_reader(clip.source_id)
            if reader is None:
                timeline_frame += frames_left_in_clip
                continue

            sub_buffer = []
            sub_lock = threading.Lock()
            local_stop = threading.Event()

            # Link to our stop event
            def check_stop():
                while not self._stop_event.is_set() and not local_stop.is_set():
                    time.sleep(0.01)
                local_stop.set()

            stop_watcher = threading.Thread(target=check_stop, daemon=True)
            stop_watcher.start()

            reader.read_sequential_into_buffer(
                source_frame, sub_buffer, frames_left_in_clip,
                local_stop, sub_lock,
            )
            local_stop.set()

            # Transfer from sub_buffer to main buffer
            for _, frame_data in sub_buffer:
                if self._stop_event.is_set():
                    return

                # Wait if main buffer is full
                while not self._stop_event.is_set():
                    with self._buffer_lock:
                        if len(self._buffer) < PREFETCH_BUFFER_SIZE:
                            self._buffer.append((timeline_frame, frame_data))
                            break
                    self._stop_event.wait(0.005)

                timeline_frame += 1
                if timeline_frame >= total:
                    return

            # If sub_buffer was shorter than expected (decode error), skip ahead
            if len(sub_buffer) < frames_left_in_clip:
                timeline_frame = clip_start + clip.duration_frames


class ScrubDecoder(QObject):
    """Handles asynchronous frame decoding for scrubbing.
    Debounces rapid requests and decodes in a background thread."""

    frame_decoded = Signal(int, np.ndarray)  # (timeline_frame, frame_data)
    _restart_timer = Signal()  # internal: safely restart debounce timer from any thread

    def __init__(self, timeline: TimelineModel, sources: Dict[str, VideoSource],
                 reader_pool: VideoReaderPool, proxy_manager: Optional[ProxyManager] = None,
                 parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._reader_pool = reader_pool
        self._proxy_manager = proxy_manager

        self._pending_frame: Optional[int] = None
        self._lock = threading.Lock()
        self._decode_thread: Optional[threading.Thread] = None
        self._busy = False

        # Direction tracking for smart pre-decode
        self._last_requested_frame: Optional[int] = None
        self._scrub_direction = 1  # +1 forward, -1 backward

        # Debounce timer — coalesce rapid scrub requests
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(8)  # ~120Hz max decode rate
        self._debounce_timer.timeout.connect(self._dispatch_decode)
        self._restart_timer.connect(self._debounce_timer.start)

    def set_proxy_manager(self, proxy_manager: ProxyManager):
        self._proxy_manager = proxy_manager

    def request_frame(self, timeline_frame: int):
        """Request a frame. Prefers full-res cached frames; proxy is last resort."""
        result = self._timeline.timeline_frame_to_source_frame(timeline_frame)
        if result is None:
            return
        clip, source_frame = result
        cache = get_frame_cache()

        # Track scrub direction for pre-decode
        if self._last_requested_frame is not None:
            delta = timeline_frame - self._last_requested_frame
            if delta > 0:
                self._scrub_direction = 1
            elif delta < 0:
                self._scrub_direction = -1
        self._last_requested_frame = timeline_frame

        # 1. Exact full-res cache hit — instant, best quality
        cached = cache.get(clip.source_id, source_frame)
        if cached is not None:
            self.frame_decoded.emit(timeline_frame, cached)
            return

        # 2. Nearest full-res cached frame — still full quality, slightly off
        nearest = cache.get_nearest(clip.source_id, source_frame, tolerance=200)
        if nearest is not None:
            _, nearest_frame = nearest
            self.frame_decoded.emit(timeline_frame, nearest_frame)
            # Queue exact frame decode in background
            with self._lock:
                self._pending_frame = timeline_frame
            if not self._debounce_timer.isActive():
                self._debounce_timer.start()
            return

        # 3. Queue full-res decode in background (always)
        with self._lock:
            self._pending_frame = timeline_frame
        if not self._debounce_timer.isActive():
            self._debounce_timer.start()

    def _dispatch_decode(self):
        """Called after debounce — start background decode of latest requested frame."""
        with self._lock:
            frame = self._pending_frame
            self._pending_frame = None
            if frame is None or self._busy:
                return
            self._busy = True

        thread = threading.Thread(target=self._decode_worker, args=(frame,), daemon=True)
        thread.start()

    def _decode_worker(self, timeline_frame: int):
        """Background thread: decode the requested frame, then pre-decode nearby frames."""
        try:
            # Check if a newer request already arrived — skip this stale one
            with self._lock:
                if self._pending_frame is not None:
                    latest = self._pending_frame
                    self._pending_frame = None
                    timeline_frame = latest

            result = self._timeline.timeline_frame_to_source_frame(timeline_frame)
            if result is None:
                return
            clip, source_frame = result

            cache = get_frame_cache()
            cached = cache.get(clip.source_id, source_frame)
            if cached is not None:
                self.frame_decoded.emit(timeline_frame, cached)
            else:
                reader = self._reader_pool.get_reader(clip.source_id)
                if reader is None:
                    return
                frame_data = reader.seek_frame(source_frame)
                if frame_data is not None:
                    self.frame_decoded.emit(timeline_frame, frame_data)

            # Pre-decode ahead in the scrub direction. Sequential decode is
            # 100-200x faster than random seeking, so this fills the cache
            # quickly and makes the next scrub movements instant cache hits.
            self._predecode_directional(timeline_frame, self._scrub_direction)

        except Exception as e:
            logger.debug("Scrub decode failed for frame %d: %s", timeline_frame, e)
        finally:
            with self._lock:
                self._busy = False
                if self._pending_frame is not None:
                    latest = self._pending_frame
                    self._pending_frame = None
                    self._busy = True
                    thread = threading.Thread(
                        target=self._decode_worker, args=(latest,), daemon=True
                    )
                    thread.start()

    def _predecode_directional(self, center_frame: int, direction: int):
        """Pre-decode frames ahead of the scrub direction into the cache.
        90 frames in the primary direction, 10 in reverse.
        Stops immediately if a new scrub request arrives."""
        cache = get_frame_cache()
        total = self._timeline.get_total_duration_frames()

        primary_count = 90
        reverse_count = 10

        # Primary direction first (where user is scrubbing toward)
        for offset in range(1, primary_count + 1):
            with self._lock:
                if self._pending_frame is not None:
                    return

            frame = center_frame + offset * direction
            if frame < 0 or frame >= total:
                break

            result = self._timeline.timeline_frame_to_source_frame(frame)
            if result is None:
                continue
            clip, source_frame = result

            if cache.get(clip.source_id, source_frame) is not None:
                continue

            reader = self._reader_pool.get_reader(clip.source_id)
            if reader is None:
                continue
            try:
                reader.seek_frame(source_frame)
            except Exception:
                pass

        # Small reverse buffer for direction changes
        for offset in range(1, reverse_count + 1):
            with self._lock:
                if self._pending_frame is not None:
                    return

            frame = center_frame - offset * direction
            if frame < 0 or frame >= total:
                break

            result = self._timeline.timeline_frame_to_source_frame(frame)
            if result is None:
                continue
            clip, source_frame = result

            if cache.get(clip.source_id, source_frame) is not None:
                continue

            reader = self._reader_pool.get_reader(clip.source_id)
            if reader is None:
                continue
            try:
                reader.seek_frame(source_frame)
            except Exception:
                pass
