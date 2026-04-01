import logging
import os

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, Signal

# Ensure libmpv DLL is findable
os.environ['PATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ['PATH']
import mpv

logger = logging.getLogger(__name__)


class PreviewWidget(QWidget):
    """Video preview using mpv for GPU-accelerated decode and display.

    mpv handles the entire pipeline on GPU: NVDEC decode → GPU buffer → display.
    No frames touch CPU RAM during scrubbing."""

    seek_complete = Signal()  # emitted when mpv finishes seeking

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMinimumSize(320, 180)

        # Container widget for mpv to render into
        self._container = QWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)

        self._player: mpv.MPV = None
        self._current_source: str = None
        self._ready = False

    def init_player(self):
        """Initialize the mpv player. Must be called after the widget is shown
        (so winId() is valid)."""
        if self._player is not None:
            return
        wid = str(int(self._container.winId()))
        self._player = mpv.MPV(
            wid=wid,
            hwdec='auto',          # NVDEC when available
            hr_seek='yes',         # frame-accurate seeking
            keep_open='yes',       # don't close at EOF
            keep_open_pause='yes',
            osd_level=0,           # no OSD
            cursor_autohide='no',
            input_cursor='no',
            input_default_bindings='no',
            input_vo_keyboard='no',
            ao='null',             # no audio output
            video_sync='display-resample',
        )
        self._player.pause = True
        self._ready = True
        logger.info("mpv player initialized (hwdec=%s, wid=%s)", 'auto', wid)

    def load_source(self, file_path: str):
        """Load a video source. Fast if already loaded."""
        if not self._ready:
            self.init_player()
        if self._current_source == file_path:
            return
        self._player.pause = True
        self._player.loadfile(file_path)
        self._player.wait_for_property('seekable')
        self._current_source = file_path
        logger.debug("Loaded source: %s", file_path)

    def seek_to_time(self, timestamp: float):
        """Seek to an exact timestamp (seconds). GPU-accelerated."""
        if not self._ready or self._current_source is None:
            return
        self._player.command('seek', str(timestamp), 'absolute+exact')

    def seek_to_frame(self, frame_number: int, fps: float):
        """Seek to an exact frame number."""
        if fps > 0:
            self.seek_to_time(frame_number / fps)

    def play(self):
        """Start playback."""
        if self._ready:
            self._player.pause = False

    def pause(self):
        """Pause playback."""
        if self._ready:
            self._player.pause = True

    @property
    def is_playing(self) -> bool:
        if self._ready and self._player:
            return not self._player.pause
        return False

    def show_frame(self, frame: np.ndarray, fast: bool = True):
        """Legacy compatibility: display a numpy frame.
        Only used for gaps or when mpv isn't available."""
        # For gap display, we could clear the screen or show black
        pass

    def clear_frame(self):
        """Show black (gap on timeline)."""
        if self._ready:
            self._player.command('stop')
            self._current_source = None

    def cleanup(self):
        """Clean up mpv player."""
        if self._player is not None:
            try:
                self._player.terminate()
            except Exception:
                pass
            self._player = None
            self._ready = False

    def __del__(self):
        self.cleanup()
