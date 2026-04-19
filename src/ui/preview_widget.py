import logging
import math
import os
import time

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QComboBox
from PySide6.QtCore import Qt, Signal, QTimer

# Ensure libmpv DLL is findable
os.environ['PATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ['PATH']
import mpv

logger = logging.getLogger(__name__)


class PreviewWidget(QWidget):
    """Video preview using mpv for GPU-accelerated decode and display.

    mpv handles the entire pipeline on GPU: NVDEC decode → GPU buffer → display.
    No frames touch CPU RAM during scrubbing."""

    seek_complete = Signal()  # emitted when mpv finishes seeking
    zoom_changed = Signal(str)  # emitted when zoom mode/level changes (display text)

    # Zoom presets shown in the dropdown
    _ZOOM_PRESETS = ["Fit", "25%", "50%", "75%", "100%", "150%", "200%", "400%"]
    _ZOOM_MIN_PERCENT = 10.0
    _ZOOM_MAX_PERCENT = 1600.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMinimumSize(320, 180)

        # Black overlay for gaps — sits on top of mpv, hidden by default
        self._black_overlay = QWidget(self)
        self._black_overlay.setStyleSheet("background-color: #1a1a1a;")
        self._black_overlay.hide()

        # Container widget for mpv to render into
        self._container = QWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)

        self._player: mpv.MPV = None
        self._current_source: str = None
        self._ready = False
        self._is_playing = False  # cached to avoid querying mpv C property
        self._showing_black = False
        self._last_seek_time = 0.0
        self._min_seek_interval = 0.033  # ~30fps max seek rate
        self._scrubbing = False          # True during active playhead drag

        # Zoom / pan state (session-only, persists across clips)
        self._zoom_mode = "fit"         # "fit" or "percent"
        self._zoom_percent = 100.0      # only used when mode == "percent"
        self._pan_x = 0.0               # mpv video-pan-x (fraction of window width)
        self._pan_y = 0.0
        self._source_w = 0              # native source dims (captured in load_source)
        self._source_h = 0
        self._panning = False
        self._pan_start_mouse = None
        self._pan_start_offset = (0.0, 0.0)
        self._updating_combo = False

        # Bottom-left zoom dropdown overlay
        # Kept editable so wheel zoom can display non-preset values like "137%",
        # but the lineEdit is read-only and clicking anywhere on it opens the popup.
        self._zoom_combo = QComboBox(self)
        self._zoom_combo.setEditable(True)
        self._zoom_combo.addItems(self._ZOOM_PRESETS)
        self._zoom_combo.setCurrentText("Fit")
        self._zoom_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        line_edit = self._zoom_combo.lineEdit()
        line_edit.setReadOnly(True)
        line_edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        line_edit.setCursor(Qt.CursorShape.ArrowCursor)
        line_edit.installEventFilter(self)
        # Watch the popup's hide event so clicking the combo while the popup
        # is open closes it instead of immediately reopening it.
        self._zoom_combo.view().installEventFilter(self)
        self._zoom_popup_closed_at = 0.0
        self._zoom_combo.setStyleSheet(
            "QComboBox {"
            " background-color: #3a3a3a;"
            " color: #ddd;"
            " border: 1px solid #555;"
            " border-radius: 3px;"
            " padding: 2px 6px;"
            "}"
            "QComboBox:hover {"
            " background-color: #444;"
            "}"
            "QComboBox QAbstractItemView {"
            " background-color: #3a3a3a;"
            " color: #ddd;"
            " border: 1px solid #555;"
            " selection-background-color: #5577aa;"
            " selection-color: #fff;"
            " outline: 0;"
            "}"
            "QComboBox QLineEdit {"
            " background: transparent;"
            " color: #ddd;"
            " border: none;"
            " padding: 0;"
            " selection-background-color: transparent;"
            "}"
        )
        self._zoom_combo.setFixedWidth(90)
        self._zoom_combo.activated.connect(self._on_combo_activated)
        self._zoom_combo.raise_()

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
            hr_seek_framedrop='yes',  # skip unnecessary frames during seek decode
            keep_open='yes',       # don't close at EOF
            keep_open_pause='yes',
            osd_level=0,           # no OSD
            cursor_autohide='no',
            input_cursor='no',
            input_default_bindings='no',
            input_vo_keyboard='no',
            ao='null',             # no audio output
            video_sync='display-resample',
            demuxer_max_bytes=str(150 * 1024 * 1024),     # 150MB forward cache
            demuxer_max_back_bytes=str(75 * 1024 * 1024),  # 75MB backward cache
        )
        self._player.pause = True
        self._is_playing = False
        self._ready = True
        logger.info("mpv player initialized (hwdec=%s, wid=%s)", 'auto', wid)

    def load_source(self, file_path: str):
        """Load a video source. Fast if already loaded."""
        if not self._ready:
            self.init_player()
        if self._current_source == file_path:
            return
        self._ensure_video_visible()
        self._player.pause = True
        self._is_playing = False
        self._player.loadfile(file_path, 'replace')
        # Wait for file to become seekable before allowing seeks
        try:
            self._player.wait_for_property('seekable')
        except Exception:
            pass
        self._current_source = file_path
        self._showing_black = False
        # Capture native source dimensions for zoom math (may be None briefly)
        try:
            w = self._player.width
            h = self._player.height
            if w and h:
                self._source_w = int(w)
                self._source_h = int(h)
        except Exception:
            pass
        # Re-project current zoom mode onto the new source
        self._apply_zoom()
        logger.debug("Loaded source: %s", file_path)

    def get_time_pos(self) -> float:
        """Get mpv's current playback position in seconds, or -1."""
        if self._ready and self._player:
            try:
                t = self._player.time_pos
                return t if t is not None else -1.0
            except Exception:
                pass
        return -1.0

    def show_black(self):
        """Show black screen for gaps. No mpv state change — just overlay."""
        self._black_overlay.setGeometry(self._container.geometry())
        self._black_overlay.raise_()
        self._black_overlay.show()
        # Keep the zoom combo clickable above the overlay
        self._zoom_combo.raise_()

    def scrub_start(self):
        """Called when the user begins dragging the playhead."""
        self._scrubbing = True

    def scrub_end(self):
        """Called when the user releases the playhead drag."""
        self._scrubbing = False

    def seek_to_time(self, timestamp: float):
        """Seek to an exact timestamp (seconds). GPU-accelerated.
        Uses async seeks so the Qt main thread never blocks on mpv decode.
        Throttled to ~30fps to prevent flooding mpv's command queue."""
        if not self._ready or self._current_source is None:
            return
        now = time.monotonic()
        if now - self._last_seek_time < self._min_seek_interval:
            return
        self._last_seek_time = now
        self._do_seek(timestamp)

    def _do_seek(self, timestamp: float):
        """Execute a non-blocking mpv seek. mpv decodes the target frame
        in the background and displays it when ready. If a new seek arrives
        before the previous completes, mpv abandons the old one."""
        self._ensure_video_visible()
        hiding_overlay = self._black_overlay.isVisible()
        try:
            self._player.command_async('seek', str(timestamp), 'absolute+exact')
        except Exception:
            return
        if hiding_overlay:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, self._black_overlay.hide)
        else:
            self._black_overlay.hide()

    def seek_to_frame(self, frame_number: int, fps: float):
        """Seek to an exact frame number."""
        if fps > 0:
            self.seek_to_time(frame_number / fps)

    def play(self):
        """Start playback."""
        if self._ready:
            self._player.pause = False
            self._is_playing = True

    def pause(self):
        """Pause playback."""
        if self._ready:
            self._player.pause = True
            self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def show_frame(self, frame: np.ndarray, fast: bool = True):
        """Legacy compatibility: display a numpy frame.
        Only used for gaps or when mpv isn't available."""
        pass

    def clear_frame(self):
        """Show black (gap on timeline). Keeps the source loaded for fast resume."""
        if self._ready and self._current_source:
            self._player.vid = 'no'
            self._showing_black = True

    def _ensure_video_visible(self):
        """Re-enable video track if it was hidden for a gap."""
        if getattr(self, '_showing_black', False):
            self._player.vid = 'auto'
            self._showing_black = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._black_overlay.isVisible():
            self._black_overlay.setGeometry(self._container.geometry())
        # Position zoom combo in bottom-left of the container
        self._reposition_zoom_combo()
        # Fit-factor changes with widget size, so a fixed percentage needs to
        # be re-translated into mpv's video-zoom.
        if self._zoom_mode == "percent":
            self._apply_zoom()

    def _reposition_zoom_combo(self):
        if not hasattr(self, "_zoom_combo"):
            return
        inset = 8
        geo = self._container.geometry()
        combo_h = self._zoom_combo.sizeHint().height()
        combo_w = self._zoom_combo.width()
        x = geo.x() + inset
        y = geo.y() + geo.height() - combo_h - inset
        self._zoom_combo.setGeometry(x, y, combo_w, combo_h)
        self._zoom_combo.raise_()

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------

    def _get_render_geometry(self):
        """Return (widget_w, widget_h, fit_w, fit_h, r, scaled_w, scaled_h) or None.

        - fit_w/fit_h: pixel size of the video at video_zoom=0 (fit-to-window)
        - r: current zoom ratio over fit (1.0 = fit)
        - scaled_w/scaled_h: actual rendered pixel size
        mpv's video-pan-x/y are in fractions of scaled_w/scaled_h.
        """
        if self._source_w <= 0 or self._source_h <= 0:
            return None
        widget_w = max(1, self._container.width())
        widget_h = max(1, self._container.height())
        aspect = self._source_w / self._source_h
        fit_w = min(widget_w, widget_h * aspect)
        fit_h = min(widget_h, widget_w / aspect)
        if fit_w <= 0 or fit_h <= 0:
            return None
        if self._zoom_mode == "fit":
            r = 1.0
        else:
            fit_factor = fit_w / self._source_w
            if fit_factor <= 0:
                return None
            r = (self._zoom_percent / 100.0) / fit_factor
            if r <= 0:
                return None
        return (widget_w, widget_h, fit_w, fit_h, r, r * fit_w, r * fit_h)

    def _apply_zoom(self):
        """Push current zoom_mode/percent/pan into mpv properties."""
        if not self._ready or self._player is None:
            return
        # Opportunistically refresh source dims if we don't have them yet
        if self._source_w <= 0 or self._source_h <= 0:
            try:
                w = self._player.width
                h = self._player.height
                if w and h:
                    self._source_w = int(w)
                    self._source_h = int(h)
            except Exception:
                pass
        try:
            if self._zoom_mode == "fit":
                self._player.video_zoom = 0.0
                self._player.video_pan_x = 0.0
                self._player.video_pan_y = 0.0
                self._pan_x = 0.0
                self._pan_y = 0.0
                self._refresh_combo_text("Fit")
                return

            geo = self._get_render_geometry()
            if geo is None:
                # Source dims not yet known — defer, leave state as-is
                return
            widget_w, widget_h, _fit_w, _fit_h, r, scaled_w, scaled_h = geo
            video_zoom = math.log2(r)
            self._clamp_pan(widget_w, widget_h, scaled_w, scaled_h)
            self._player.video_zoom = video_zoom
            self._player.video_pan_x = self._pan_x
            self._player.video_pan_y = self._pan_y
            self._refresh_combo_text(f"{int(round(self._zoom_percent))}%")
        except Exception as e:
            logger.debug("apply_zoom failed: %s", e)

    def _clamp_pan(self, widget_w, widget_h, scaled_w, scaled_h):
        """Clamp pan so the scaled video still covers the widget.

        mpv's video-pan-x/y are in fractions of the scaled video size. With the
        video centered at pan=0, the max pan that keeps the video edge flush
        with the widget edge is 0.5 * (1 - widget/scaled). When the scaled
        video is smaller than the widget on an axis, no pan is meaningful.
        """
        if scaled_w > widget_w:
            max_px = 0.5 * (1.0 - widget_w / scaled_w)
            if self._pan_x > max_px:
                self._pan_x = max_px
            elif self._pan_x < -max_px:
                self._pan_x = -max_px
        else:
            self._pan_x = 0.0
        if scaled_h > widget_h:
            max_py = 0.5 * (1.0 - widget_h / scaled_h)
            if self._pan_y > max_py:
                self._pan_y = max_py
            elif self._pan_y < -max_py:
                self._pan_y = -max_py
        else:
            self._pan_y = 0.0

    def _refresh_combo_text(self, text: str):
        if not hasattr(self, "_zoom_combo"):
            return
        if self._zoom_combo.currentText() == text:
            return
        self._updating_combo = True
        try:
            self._zoom_combo.setCurrentText(text)
        finally:
            self._updating_combo = False
        self.zoom_changed.emit(text)

    def _on_combo_activated(self, index: int):
        if self._updating_combo:
            return
        text = self._zoom_combo.itemText(index)
        self._set_zoom_from_text(text)

    def eventFilter(self, obj, event):
        # Popup just closed — record timestamp so a click on the combo within
        # the same gesture doesn't immediately reopen it.
        if obj is self._zoom_combo.view() and event.type() == event.Type.Hide:
            self._zoom_popup_closed_at = time.monotonic()
            return super().eventFilter(obj, event)
        # Click anywhere on the read-only lineEdit opens the dropdown popup —
        # unless the popup was just dismissed by the same click (toggle-close).
        if obj is self._zoom_combo.lineEdit() and event.type() == event.Type.MouseButtonPress:
            if time.monotonic() - self._zoom_popup_closed_at < 0.2:
                return True
            self._zoom_combo.showPopup()
            return True
        return super().eventFilter(obj, event)

    def _set_zoom_from_text(self, text: str):
        t = text.strip().lower().rstrip("%").strip()
        if t == "fit" or t == "":
            self._zoom_mode = "fit"
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._apply_zoom()
            return
        try:
            pct = float(t)
        except ValueError:
            # Not a number — snap display back to whatever the state says
            self._apply_zoom()
            return
        pct = max(self._ZOOM_MIN_PERCENT, min(self._ZOOM_MAX_PERCENT, pct))
        self._zoom_mode = "percent"
        self._zoom_percent = pct
        self._apply_zoom()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        # Only zoom when the wheel is over the video area, not over the combo
        pos = event.position().toPoint()
        if self._zoom_combo.geometry().contains(pos):
            super().wheelEvent(event)
            return
        if self._source_w <= 0 or self._source_h <= 0:
            super().wheelEvent(event)
            return

        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return

        widget_w = max(1, self._container.width())
        widget_h = max(1, self._container.height())
        aspect = self._source_w / self._source_h
        fit_w = min(widget_w, widget_h * aspect)
        fit_h = min(widget_h, widget_w / aspect)
        if fit_w <= 0 or fit_h <= 0:
            return
        fit_factor = fit_w / self._source_w

        # Current zoom ratio over fit (r = 1 means we're at fit)
        if self._zoom_mode == "fit":
            r_old = 1.0
            old_percent = fit_factor * 100.0
        else:
            r_old = (self._zoom_percent / 100.0) / fit_factor
            old_percent = self._zoom_percent

        new_percent = old_percent * (1.25 ** steps)
        new_percent = max(self._ZOOM_MIN_PERCENT, min(self._ZOOM_MAX_PERCENT, new_percent))
        if new_percent == old_percent:
            return
        r_new = (new_percent / 100.0) / fit_factor

        # Zoom toward cursor: keep the source point under the mouse fixed.
        # Screen position of source point u' (in [-0.5, 0.5] source-fraction):
        #     screen_x = W/2 + (pan_x + u') * scaled_w    where scaled_w = r * fit_w
        # Solving for pan_x such that the same u' stays under the same cursor:
        #     pan_x_new = pan_x_old + (dx_cursor / fit_w) * (1/r_new - 1/r_old)
        dx_cursor = pos.x() - widget_w / 2.0
        dy_cursor = pos.y() - widget_h / 2.0
        delta = (1.0 / r_new) - (1.0 / r_old)
        self._pan_x = self._pan_x + (dx_cursor / fit_w) * delta
        self._pan_y = self._pan_y + (dy_cursor / fit_h) * delta
        self._zoom_mode = "percent"
        self._zoom_percent = new_percent
        self._apply_zoom()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._zoom_mode == "percent":
            self._panning = True
            self._pan_start_mouse = event.position().toPoint()
            self._pan_start_offset = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            pos = event.position().toPoint()
            dx = pos.x() - self._pan_start_mouse.x()
            dy = pos.y() - self._pan_start_mouse.y()
            geo = self._get_render_geometry()
            if geo is None:
                return
            _ww, _wh, _fw, _fh, _r, scaled_w, scaled_h = geo
            # mpv pan is in fractions of the scaled video size: moving the
            # video by dx screen pixels requires dx/scaled_w pan-x units.
            # This makes the grabbed pixel track the cursor exactly.
            start_px, start_py = self._pan_start_offset
            self._pan_x = start_px + dx / scaled_w
            self._pan_y = start_py + dy / scaled_h
            self._apply_zoom()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Don't hijack double-clicks on the combo
        pos = event.position().toPoint()
        if self._zoom_combo.geometry().contains(pos):
            super().mouseDoubleClickEvent(event)
            return
        self._zoom_mode = "fit"
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._apply_zoom()
        event.accept()

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
