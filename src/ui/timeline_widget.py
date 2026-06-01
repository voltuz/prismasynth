import logging
import os
from enum import IntEnum
from typing import Optional, List

from PySide6.QtWidgets import QWidget, QScrollBar, QVBoxLayout, QToolButton, QMenu
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QAction, QPolygon,
    QMouseEvent, QWheelEvent, QPaintEvent, QKeyEvent,
    QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent,
)

from core.timeline import TimelineModel
from core.clip import Clip
from core.crop_region import clamp_anchor, required_source_frames
from core.ui_scale import ui_scale
from ui.icon_loader import icon

# Custom MIMEs for media-pool drags (kept in sync with ui/media_panel.py).
_SOURCE_ID_MIME = "application/x-prismasynth-source-ids"
_SOURCE_DURATIONS_MIME = "application/x-prismasynth-source-durations"
_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv',
                     '.flv', '.webm', '.m4v', '.ts', '.mxf'}
_DROP_INDICATOR_COLOR = QColor(255, 200, 0)
_DROP_FOOTPRINT_COLOR = QColor(255, 200, 0, 60)  # translucent fill

logger = logging.getLogger(__name__)


class EditMode(IntEnum):
    SELECTION = 0
    CUT = 1

# Clip color palette (8 alternating colors for visual distinction)
CLIP_COLORS = [
    QColor(70, 130, 180),   # steel blue
    QColor(60, 160, 120),   # sea green
    QColor(180, 100, 60),   # sienna
    QColor(140, 80, 160),   # purple
    QColor(180, 160, 50),   # olive
    QColor(80, 150, 150),   # teal
    QColor(160, 80, 80),    # indian red
    QColor(100, 130, 70),   # olive drab
]

# User-controlled clip body height — kept in absolute pixels so the
# user's manual drag preference retains its meaning when the UI scale
# changes. (Per the v0.14 UI-scale design: chrome scales, the user's
# clip-height drag stays an absolute pixel value.)
CLIP_HEIGHT_DEFAULT = 60
CLIP_HEIGHT_MIN = 30
CLIP_HEIGHT_MAX = 200
# Group-strip layout — count, not a pixel dimension.
GROUP_LABEL_MAX_VISIBLE = 3
# Design-pixel baselines for chrome metrics. Multiplied by the active
# UI scale into per-instance attributes (self.HEADER_HEIGHT, etc.) by
# TimelineStrip._refresh_scale_metrics() — paint code reads the
# instance attributes rather than these module-level baselines.
_DH_HEADER = 48
_DH_PAD = 4
_DH_RESIZE_HANDLE = 5
_DH_THUMB_HIDE = 15  # below this clip width, skip thumbnail paint
_DH_GROUP_LABEL = 14
# Crop-strip layout. Strips live below the group-chip band; each crop
# gets its own row. Strip width = the 81-frame export window's source-
# frame width at the current zoom; drag the strip to translate the
# window across the clip.
_DH_CROP_STRIP = 8
_DH_CROP_GAP = 1
MIN_CROP_STRIP_PX = 6
CROP_WARNING_COLOR = QColor("#cc5500")  # clip too short to host the window
_DH_LABEL_FONT_PT = 8
_DH_THUMB_PAD = 6    # thumbnail inset inside a clip rect
_DH_LABEL_GAP = 40   # min clip px beyond the thumbnails before the label shows
PLAYHEAD_COLOR = QColor(255, 50, 50)
SELECTION_BORDER = QColor(255, 255, 100)
GAP_COLOR = QColor(30, 30, 30)
BG_COLOR = QColor(40, 40, 40)
RULER_BG = QColor(50, 50, 50)
RULER_TEXT = QColor(180, 180, 180)


class TimelineStrip(QWidget):
    """Custom-painted timeline strip showing clips as colored blocks."""

    playhead_moved = Signal(int)   # timeline frame
    scrub_started = Signal()       # user started dragging playhead
    scrub_ended = Signal()         # user released playhead drag
    pan_started = Signal()         # user started middle-mouse pan
    pan_ended = Signal()           # user released middle-mouse pan
    scroll_changed = Signal()      # scroll offset changed (no playhead change)
    clip_clicked = Signal(str, object)  # clip_id, QMouseEvent (for modifier keys)
    preview_frame_requested = Signal(int)  # cut-mode hover scrub (no playhead move)
    cut_requested = Signal(int)    # cut-mode click at timeline frame
    sources_dropped = Signal(list, int)   # source_ids (list[str]), insert frame
    files_dropped = Signal(list, int)     # file paths (list[str]), insert frame
    crop_clicked = Signal(str, str, str)  # clip_id, crop_id, segment_id (block hit)

    def __init__(self, model: TimelineModel, parent=None):
        super().__init__(parent)
        self._model = model
        self._playhead_frame = 0
        self._pixels_per_frame = 0.5  # zoom level
        self._scroll_offset = 0       # horizontal scroll in pixels
        self._clip_height = CLIP_HEIGHT_DEFAULT
        self._dragging_playhead = False
        self._panning = False          # middle-mouse pan
        self._pan_start_x = 0
        self._pan_start_offset = 0
        self._resizing_track = False   # dragging bottom edge to resize
        self._resize_start_y = 0
        self._resize_start_height = 0
        self._marquee_active = False   # rubber band selection below track
        self._marquee_start: Optional[QPoint] = None
        self._marquee_end: Optional[QPoint] = None
        self._marquee_ctrl = False
        self._thumbnails = {}  # (clip_id, "first"|"last") -> QPixmap
        self._thumbnails_enabled = True
        self._fps = 24.0
        self._edit_mode = EditMode.SELECTION
        self._scrub_follow = False     # F key: playhead follows mouse without clicking
        self._cut_preview_x: Optional[int] = None
        self._last_preview_frame = -1   # throttle cut-mode hover seeks

        # Drop state — set during drag-over from the Media Panel; consumed by
        # paintEvent to draw a vertical insertion line and (when source
        # durations are known) a translucent footprint rectangle showing
        # how much room the dropped clip(s) will occupy.
        self._drop_insert_frame: Optional[int] = None
        self._drop_total_frames: int = 0
        self._drop_source_ids: list = []      # parallel to _drop_durations
        self._drop_durations: list = []
        self._drop_thumb_cache: dict = {}     # sid -> QPixmap (lazy-loaded)
        # Sources dict reference — set by MainWindow once at startup so the
        # drop preview can render each dragged source's media-pool thumbnail.
        self._sources_ref: dict = {}

        # Crop-strip drag — populated on press over a crop strip and
        # consumed by mouseMove/mouseRelease. Tuple: (clip_id, crop_id,
        # original_source_anchor, clip_timeline_start, required_src_frames).
        self._dragging_crop_strip: Optional[tuple] = None
        # True while the active strip drag is an Alt-clone (a temp segment
        # was appended to drag; committed as an add on release).
        self._crop_strip_clone: bool = False
        # True when the press landed on the already-selected segment — a
        # pure click (no drag) then deselects it on release.
        self._crop_strip_was_selected: bool = False
        # Track press-x so we can distinguish "click only" from "drag" and
        # apply a small dead-zone before mutating the anchor.
        self._crop_strip_press_x: float = 0.0
        # Selected crop id mirrored from main_window via set_selected_crop_id;
        # used purely for the white selection halo on the matching strip.
        self._selected_crop_id: str = ""
        # Selected segment id — drives the white halo on the matching block
        # and marks the drag/clone/delete target.
        self._selected_segment_id: str = ""

        # Materialise scaled chrome metrics BEFORE any size-dependent calls
        # below — setMinimumHeight reads self.HEADER_HEIGHT etc.
        self._refresh_scale_metrics()

        # Reserve room for the group-label strip below the clip body.
        self.setMinimumHeight(
            CLIP_HEIGHT_MIN + self.HEADER_HEIGHT
            + ui_scale().px(4) + self.GROUP_STRIP_RESERVE)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        self._model.clips_changed.connect(self.update)
        self._model.clips_changed.connect(self._refresh_min_height)
        self._model.selection_changed.connect(self.update)
        self._model.in_out_changed.connect(self.update)
        self._model.groups_changed.connect(self.update)
        ui_scale().changed.connect(self._on_ui_scale_changed)

    def _refresh_scale_metrics(self):
        """(Re)compute scaled chrome metrics from the design baselines and
        the current UI scale. Called once in __init__ and again whenever
        the user picks a new scale via the View menu."""
        s = ui_scale()
        self.HEADER_HEIGHT = s.px(_DH_HEADER)
        self.TIMELINE_H_PADDING = s.px(_DH_PAD)
        self.RESIZE_HANDLE_HEIGHT = s.px(_DH_RESIZE_HANDLE)
        self.THUMB_HIDE_THRESHOLD = s.px(_DH_THUMB_HIDE)
        self.GROUP_LABEL_HEIGHT = s.px(_DH_GROUP_LABEL)
        self.GROUP_STRIP_RESERVE = self.GROUP_LABEL_HEIGHT * GROUP_LABEL_MAX_VISIBLE
        self.CROP_STRIP_HEIGHT = s.px(_DH_CROP_STRIP)
        self.CROP_STRIP_GAP = s.px(_DH_CROP_GAP)
        self._label_font_pt = s.font_pt(_DH_LABEL_FONT_PT)
        self.THUMB_PAD = s.px(_DH_THUMB_PAD)
        self.LABEL_GAP = s.px(_DH_LABEL_GAP)

    def _on_ui_scale_changed(self):
        self._refresh_scale_metrics()
        self._refresh_min_height()
        self.update()

    def _crop_band_height(self) -> int:
        """Pixel height of the crop-strip band below the group-chip band.
        Sized to the clip with the most crops so all strips remain visible
        without per-clip height variance."""
        max_crops = 0
        for clip in self._model.clips:
            if clip.is_gap or not clip.crop_regions:
                continue
            n = len(clip.crop_regions)
            if n > max_crops:
                max_crops = n
        if max_crops == 0:
            return 0
        return max_crops * (self.CROP_STRIP_HEIGHT + self.CROP_STRIP_GAP)

    def _refresh_min_height(self):
        """Recompute the widget's minimum height when the chrome or the
        max-crops-per-clip count changes."""
        self.setMinimumHeight(
            CLIP_HEIGHT_MIN + self.HEADER_HEIGHT
            + ui_scale().px(4) + self.GROUP_STRIP_RESERVE
            + self._crop_band_height())
        self.updateGeometry()

    def set_selected_crop_id(self, crop_id: str):
        """External crop-selection sync (panel ⇄ preview ⇄ timeline)."""
        new = crop_id or ""
        if new == self._selected_crop_id:
            return
        self._selected_crop_id = new
        self.update()

    def set_selected_segment_id(self, segment_id: str):
        """External segment-selection sync (panel ⇄ timeline)."""
        new = segment_id or ""
        if new == self._selected_segment_id:
            return
        self._selected_segment_id = new
        self.update()

    @property
    def pixels_per_frame(self) -> float:
        return self._pixels_per_frame

    @property
    def playhead_frame(self) -> int:
        return self._playhead_frame

    def set_fps(self, fps: float):
        self._fps = fps if fps > 0 else 24.0

    @property
    def edit_mode(self) -> EditMode:
        return self._edit_mode

    def set_edit_mode(self, mode: EditMode):
        self._edit_mode = mode
        self._cut_preview_x = None
        self._last_preview_frame = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def toggle_scrub_follow(self):
        """Toggle scrub-follow mode: playhead follows mouse without clicking."""
        self._scrub_follow = not self._scrub_follow
        return self._scrub_follow

    def set_playhead(self, frame: int):
        total = self._model.get_total_duration_frames()
        if total == 0:
            return
        frame = max(0, min(frame, total - 1))
        if frame != self._playhead_frame:
            self._playhead_frame = frame
            self.playhead_moved.emit(frame)
            self.update()

    def set_scroll_offset(self, offset: int):
        self._scroll_offset = max(0, offset)
        self.update()
        self.scroll_changed.emit()

    def set_pixels_per_frame(self, value: float):
        """Set the timeline zoom level. Used to restore the saved zoom when
        loading a project. Bounds match the wheel-zoom clamp so a corrupt
        or out-of-range value can't render the strip degenerate."""
        new = max(0.001, min(20.0, float(value)))
        if new == self._pixels_per_frame:
            return
        self._pixels_per_frame = new
        self.update()
        # Total width changed; the parent's scrollbar needs to recompute.
        self.scroll_changed.emit()

    def set_thumbnail(self, clip_id: str, position: str, pixmap: QPixmap):
        self._thumbnails[(clip_id, position)] = pixmap
        self.update()

    def set_sources_ref(self, sources: dict):
        """Share MainWindow's source dict so drag-over previews can pull
        per-source thumbnails from disk."""
        self._sources_ref = sources

    def _get_drop_thumb(self, sid: str):
        cached = self._drop_thumb_cache.get(sid)
        if cached is not None or sid in self._drop_thumb_cache:
            return cached
        src = self._sources_ref.get(sid) if self._sources_ref else None
        pix = None
        if src is not None:
            try:
                from core.source_thumbnail import cache_path_for
                path = cache_path_for(src)
                if path.exists() and path.stat().st_size > 0:
                    p = QPixmap(str(path))
                    if not p.isNull():
                        pix = p
            except Exception:
                pix = None
        self._drop_thumb_cache[sid] = pix
        return pix

    def set_thumbnails_enabled(self, enabled: bool):
        """Toggle thumbnail rendering. When off, clips show as solid color blocks."""
        if self._thumbnails_enabled == enabled:
            return
        self._thumbnails_enabled = enabled
        if not enabled:
            self._thumbnails.clear()
        self.update()

    @property
    def thumbnails_enabled(self) -> bool:
        return self._thumbnails_enabled
        self.update()

    def get_total_width(self) -> int:
        return self._frame_to_pixel(self._model.get_total_duration_frames())

    # --- Drag and drop (sources from Media Panel, files from OS) ---

    def _accepted_drag(self, event):
        """True if the drag mime is something we want to handle. Sets the
        drop insertion frame for the cursor position as a side effect when
        the drag is accepted (so paintEvent draws the indicator line +
        translucent footprint rectangle)."""
        mime = event.mimeData()
        accept = False
        total_frames = 0
        source_ids: list = []
        durations: list = []
        if mime.hasFormat(_SOURCE_ID_MIME):
            accept = True
            id_payload = bytes(mime.data(_SOURCE_ID_MIME)).decode("utf-8", "replace")
            source_ids = [s for s in id_payload.split("\n") if s]
            # Decode the parallel durations mime so paintEvent can size the
            # ghost rectangle. Falls back to no preview rect when missing.
            if mime.hasFormat(_SOURCE_DURATIONS_MIME):
                payload = bytes(mime.data(_SOURCE_DURATIONS_MIME)).decode("utf-8", "replace")
                try:
                    durations = [int(d) for d in payload.split("\n") if d]
                    total_frames = sum(durations)
                except ValueError:
                    durations = []
                    total_frames = 0
        elif mime.hasUrls():
            for u in mime.urls():
                if u.isLocalFile():
                    ext = os.path.splitext(u.toLocalFile())[1].lower()
                    if ext in _VIDEO_EXTENSIONS:
                        accept = True
                        break
        if accept:
            self._drop_insert_frame = self._cursor_to_insert_frame(event.position().x())
            self._drop_total_frames = total_frames
            # Only the source-id drag carries per-source breakdown; OS file
            # drags fall through with empty lists (no thumbnail preview).
            if len(source_ids) == len(durations):
                self._drop_source_ids = source_ids
                self._drop_durations = durations
            else:
                self._drop_source_ids = []
                self._drop_durations = []
            self.update()
        return accept

    def _cursor_to_insert_frame(self, x: float) -> int:
        """Snap the cursor x to a clip-boundary frame when close, else the
        raw frame at the cursor."""
        target = self._pixel_to_frame(x)
        snap_px = 8  # px tolerance for snapping to clip boundaries
        cumulative = 0
        for c in self._model.clips:
            for boundary in (cumulative, cumulative + c.duration_frames):
                bx = self._frame_to_pixel(boundary) - self._scroll_offset
                if abs(bx - x) <= snap_px:
                    return boundary
            cumulative += c.duration_frames
        # Clamp to [0, total]
        total = self._model.get_total_duration_frames()
        return max(0, min(target, total))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._accepted_drag(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._accepted_drag(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self._drop_insert_frame = None
        self._drop_total_frames = 0
        self._drop_source_ids = []
        self._drop_durations = []
        self.update()

    def dropEvent(self, event: QDropEvent):
        mime = event.mimeData()
        frame = self._cursor_to_insert_frame(event.position().x())
        if mime.hasFormat(_SOURCE_ID_MIME):
            payload = bytes(mime.data(_SOURCE_ID_MIME)).decode("utf-8")
            ids = [s for s in payload.split("\n") if s]
            if ids:
                self.sources_dropped.emit(ids, frame)
                event.acceptProposedAction()
        elif mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls()
                     if u.isLocalFile()
                     and os.path.splitext(u.toLocalFile())[1].lower() in _VIDEO_EXTENSIONS]
            if paths:
                self.files_dropped.emit(paths, frame)
                event.acceptProposedAction()
        self._drop_insert_frame = None
        self._drop_total_frames = 0
        self._drop_source_ids = []
        self._drop_durations = []
        self.update()

    def ensure_playhead_visible(self):
        playhead_px = self._frame_to_pixel(self._playhead_frame) - self._scroll_offset
        margin = 100
        if playhead_px < margin:
            self._scroll_offset = max(0, self._frame_to_pixel(self._playhead_frame) - margin)
        elif playhead_px > self.width() - margin:
            self._scroll_offset = max(0, self._frame_to_pixel(self._playhead_frame) - self.width() + margin)

    # --- Painting ---

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, BG_COLOR)

        # Ruler
        self._paint_ruler(painter, w)

        # Clips
        clip_y = self.HEADER_HEIGHT + ui_scale().px(2)
        self._paint_clips(painter, clip_y, w)

        # Crop strips — one per crop, in a band below the group chips.
        # Always visible (not gated on clip selection).
        self._paint_crop_strips(painter, clip_y, w)

        # In/Out overlay
        self._paint_in_out_overlay(painter, w)

        # Playhead
        self._paint_playhead(painter, clip_y, h)

        # Cut preview line
        if self._edit_mode == EditMode.CUT and self._cut_preview_x is not None:
            self._paint_cut_preview(painter, clip_y)

        # Marquee selection rectangle (Windows Explorer style)
        if self._marquee_active and self._marquee_start and self._marquee_end:
            x1 = min(self._marquee_start.x(), self._marquee_end.x())
            y1 = min(self._marquee_start.y(), self._marquee_end.y())
            x2 = max(self._marquee_start.x(), self._marquee_end.x())
            y2 = max(self._marquee_start.y(), self._marquee_end.y())
            marquee_rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.setBrush(QBrush(QColor(0, 120, 215, 30)))
            painter.setPen(QPen(QColor(0, 120, 215, 120), 1))
            painter.drawRect(marquee_rect)

        # Drop insertion line + translucent footprint rectangle (during drag
        # from the Media Panel). The rectangle previews how much room the
        # dropped clip(s) will occupy on the timeline at the current zoom.
        if self._drop_insert_frame is not None:
            ix = self._frame_to_pixel(self._drop_insert_frame) - self._scroll_offset
            # Footprint rectangle (drawn first so the line ends up on top).
            if self._drop_total_frames > 0:
                end_x = (self._frame_to_pixel(self._drop_insert_frame
                                               + self._drop_total_frames)
                         - self._scroll_offset)
                rect_x = max(0, ix)
                rect_w = min(end_x, w) - rect_x
                if rect_w > 0:
                    painter.fillRect(rect_x, clip_y, rect_w,
                                      self._clip_height,
                                      _DROP_FOOTPRINT_COLOR)
                # Per-source thumbnails — one at the start of each source's
                # segment, mirroring the regular clip's first-frame thumbnail.
                if (self._drop_source_ids
                        and len(self._drop_source_ids) == len(self._drop_durations)):
                    pad = self.THUMB_PAD
                    th = self._clip_height - pad * 2
                    tw = int(th * 16 / 9)
                    cumulative = self._drop_insert_frame
                    for sid, dur in zip(self._drop_source_ids, self._drop_durations):
                        seg_x = self._frame_to_pixel(cumulative) - self._scroll_offset
                        seg_end_x = self._frame_to_pixel(cumulative + dur) - self._scroll_offset
                        cumulative += dur
                        seg_w = seg_end_x - seg_x
                        if seg_w <= pad * 2 or th <= 0:
                            continue
                        if seg_end_x < 0 or seg_x > w:
                            continue
                        pix = self._get_drop_thumb(sid)
                        if pix is None:
                            continue
                        painter.save()
                        thumb_clip = QRect(seg_x + pad, clip_y + pad,
                                            seg_w - pad * 2, th)
                        painter.setClipRect(thumb_clip)
                        painter.drawPixmap(seg_x + pad, clip_y + pad, tw, th, pix)
                        painter.restore()
            if 0 <= ix <= w:
                _arr = ui_scale().px(5)
                _arr_h = ui_scale().px(6)
                painter.setPen(QPen(_DROP_INDICATOR_COLOR, ui_scale().px(3)))
                painter.drawLine(ix, self.HEADER_HEIGHT, ix, h)
                painter.setBrush(QBrush(_DROP_INDICATOR_COLOR))
                painter.drawPolygon([
                    QPoint(ix - _arr, self.HEADER_HEIGHT),
                    QPoint(ix + _arr, self.HEADER_HEIGHT),
                    QPoint(ix, self.HEADER_HEIGHT + _arr_h),
                ])

        painter.end()

    def _paint_ruler(self, painter: QPainter, width: int):
        painter.fillRect(0, 0, width, self.HEADER_HEIGHT, RULER_BG)
        painter.setPen(QPen(RULER_TEXT))
        font = QFont("Segoe UI", self._label_font_pt)
        painter.setFont(font)

        # Compute tick interval based on zoom
        seconds_per_pixel = 1.0 / (self._pixels_per_frame * self._fps) if self._fps > 0 else 1.0
        # Target roughly 80 pixels between major ticks
        target_seconds = seconds_per_pixel * 80
        # Snap to nice intervals. Fallback to the largest entry when target_seconds
        # exceeds them all — without this, deep zoom-out smears thousands of 0.5s
        # ticks into a solid bar across the ruler.
        nice_intervals = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
                          1200, 1800, 3600, 7200, 14400]
        interval = nice_intervals[-1]
        for ni in nice_intervals:
            if ni >= target_seconds:
                interval = ni
                break

        start_sec = max(0, self._scroll_offset - self.TIMELINE_H_PADDING) / (self._pixels_per_frame * self._fps)
        end_sec = max(0, self._scroll_offset + width - self.TIMELINE_H_PADDING) / (self._pixels_per_frame * self._fps)
        t = (int(start_sec / interval)) * interval
        while t <= end_sec:
            x = self._frame_to_pixel(int(t * self._fps)) - self._scroll_offset
            if 0 <= x <= width:
                _tick = ui_scale().px(6)
                painter.drawLine(x, self.HEADER_HEIGHT - _tick, x, self.HEADER_HEIGHT)
                ti = int(t)
                if interval >= 3600:
                    label = f"{ti // 3600}:{(ti % 3600) // 60:02d}:{ti % 60:02d}"
                else:
                    label = f"{ti // 60}:{ti % 60:02d}"
                painter.drawText(
                    x + ui_scale().px(3),
                    self.HEADER_HEIGHT - ui_scale().px(8), label)
            t += interval

    def _frame_to_pixel(self, frame: int) -> int:
        """Convert a timeline frame number to pixel position. Single source of truth."""
        return self.TIMELINE_H_PADDING + int(frame * self._pixels_per_frame)

    def _pixel_to_frame(self, screen_x: float) -> int:
        """Convert a screen x coordinate to timeline frame number.
        Clamps to content area so the playhead can't enter the padding."""
        # Clamp screen_x to content bounds
        x = max(self.TIMELINE_H_PADDING, screen_x)
        total = self._model.get_total_duration_frames()
        if total > 0:
            max_px = self._frame_to_pixel(total - 1) - self._scroll_offset
            x = min(x, max_px)
        return max(0, int((x + self._scroll_offset - self.TIMELINE_H_PADDING)
                          / self._pixels_per_frame))

    def get_visible_clip_ids(self):
        """Return IDs of non-gap clips currently visible in the viewport.

        Clips narrower than self.THUMB_HIDE_THRESHOLD are excluded so the thumbnail
        cache doesn't burn workers generating frames that won't be painted.
        """
        visible = []
        cumulative = 0
        vw = self.width()
        for clip in self._model.clips:
            start_px = self._frame_to_pixel(cumulative)
            cumulative += clip.duration_frames
            end_px = self._frame_to_pixel(cumulative)
            clip_w = end_px - start_px
            screen_x = start_px - self._scroll_offset
            if screen_x + clip_w < 0 or screen_x > vw:
                continue
            if clip_w < self.THUMB_HIDE_THRESHOLD:
                continue
            if not clip.is_gap:
                visible.append(clip.id)
        return visible

    def _paint_clips(self, painter: QPainter, y: int, viewport_width: int):
        cumulative_frames = 0
        selected = self._model.selected_ids
        for clip in self._model.clips:
            # Compute pixel positions from cumulative frame counts
            clip_start_px = self._frame_to_pixel(cumulative_frames)
            cumulative_frames += clip.duration_frames
            clip_end_px = self._frame_to_pixel(cumulative_frames)
            clip_w = clip_end_px - clip_start_px

            screen_x = clip_start_px - self._scroll_offset

            # Skip items entirely outside viewport
            if screen_x + clip_w < 0 or screen_x > viewport_width:
                continue

            rect = QRect(screen_x, y, clip_w, self._clip_height)

            if clip.is_gap:
                # Paint gap as dark empty space with dashed border
                painter.fillRect(rect, QBrush(GAP_COLOR))
                pen = QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect)

                # Selection highlight on gap
                if clip.id in selected:
                    painter.setPen(QPen(SELECTION_BORDER, 2))
                    painter.drawRect(rect.adjusted(1, 1, -1, -1))
                continue

            # --- Real clip ---
            color = CLIP_COLORS[clip.color_index % len(CLIP_COLORS)]

            # Fill clip body
            painter.fillRect(rect, QBrush(color))

            # Selection highlight
            if clip.id in selected:
                painter.setPen(QPen(SELECTION_BORDER, 2))
                painter.drawRect(rect.adjusted(1, 1, -1, -1))

            # Clip border
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.drawRect(rect)

            # Thumbnail size: fill track height with color border, maintain 16:9 aspect
            pad = self.THUMB_PAD
            th = self._clip_height - pad * 2
            tw = int(th * 16 / 9)

            if self._thumbnails_enabled and clip_w >= self.THUMB_HIDE_THRESHOLD:
                # First frame thumbnail — always shown, cropped to clip bounds with border
                thumb_first = self._thumbnails.get((clip.id, "first"))
                if thumb_first and clip_w > pad * 2 and th > 0:
                    painter.save()
                    thumb_clip = QRect(screen_x + pad, y + pad, clip_w - pad * 2, th)
                    painter.setClipRect(thumb_clip)
                    painter.drawPixmap(screen_x + pad, y + pad, tw, th, thumb_first)
                    painter.restore()

                # Last frame thumbnail (right edge) — only when clip fits both
                thumb_last = self._thumbnails.get((clip.id, "last"))
                if thumb_last and clip_w > tw * 2 + pad * 4 and th > 0:
                    painter.drawPixmap(screen_x + clip_w - tw - pad, y + pad, tw, th, thumb_last)

            # Duration label in center
            if clip_w > tw * 2 + self.LABEL_GAP:
                painter.setPen(QPen(QColor(240, 240, 240)))
                font = QFont("Segoe UI", self._label_font_pt)
                painter.setFont(font)
                dur_secs = clip.duration_frames / self._fps if self._fps > 0 else 0
                if dur_secs >= 60:
                    label = f"{int(dur_secs)//60}:{int(dur_secs)%60:02d}"
                else:
                    label = f"{dur_secs:.1f}s"
                text_rect = QRect(screen_x + tw, y, clip_w - tw * 2, self._clip_height)
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

            # --- Group / People labels below the clip ---
            self._paint_group_labels(
                painter, clip, screen_x, y + self._clip_height + 1, clip_w)

    def _paint_group_labels(self, painter: QPainter, clip: Clip,
                             x: int, y: int, w: int):
        """Render up to GROUP_LABEL_MAX_VISIBLE group chips beneath a clip.
        Each chip is filled with the group's colour and shows the group's
        name in a high-contrast text colour. If the clip belongs to more
        groups than fit, the last visible chip overlays a '+N' indicator."""
        if not clip.group_ids:
            return
        groups = self._model.groups
        # Resolve, then sort by digit (1-9, 0, then unkeyed by name) so chip
        # order is stable regardless of the toggle order the user pressed.
        my_groups = [groups[gid] for gid in clip.group_ids if gid in groups]
        if not my_groups:
            return
        my_groups.sort(key=self._group_sort_key)
        visible = my_groups[:GROUP_LABEL_MAX_VISIBLE]
        extra = len(my_groups) - len(visible)
        font = QFont("Segoe UI", self._label_font_pt)
        painter.setFont(font)
        for i, g in enumerate(visible):
            ly = y + i * self.GROUP_LABEL_HEIGHT
            rect = QRect(x, ly, w, self.GROUP_LABEL_HEIGHT - 1)
            painter.fillRect(rect, QColor(g.color))
            # Name on the chip — high-contrast text colour.
            text_color = self._readable_text_for(g.color)
            painter.setPen(QPen(text_color))
            label_text = g.name
            # On the last visible chip, append '+N' when more groups exist.
            if extra > 0 and i == len(visible) - 1:
                label_text = f"{g.name}  +{extra}"
            painter.drawText(
                rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label_text)

    @staticmethod
    def _group_sort_key(g):
        # Keyboard-row order: 1,2,...,9,0, then unkeyed groups (by name).
        d = g.digit
        if d is None:
            primary = 11
        elif d == 0:
            primary = 10
        else:
            primary = d
        return (primary, g.name.casefold())

    @staticmethod
    def _readable_text_for(hex_str: str) -> QColor:
        """Pick black or white based on the relative luminance of ``hex_str``.
        Standard W3C contrast rule, threshold tuned to keep mid-tones legible."""
        c = QColor(hex_str)
        lum = (0.2126 * c.redF()
               + 0.7152 * c.greenF()
               + 0.0722 * c.blueF())
        return QColor(0, 0, 0) if lum > 0.55 else QColor(255, 255, 255)

    def _paint_cut_preview(self, painter: QPainter, clip_y: int):
        x = self._cut_preview_x
        if x is None or x < 0 or x > self.width():
            return
        pen = QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(x, clip_y, x, clip_y + self._clip_height)

    def _paint_in_out_overlay(self, painter: QPainter, viewport_width: int):
        in_pt = self._model.in_point
        out_pt = self._model.out_point
        if in_pt is None and out_pt is None:
            return

        total_h = self.height()
        dim_color = QColor(0, 0, 0, 120)
        marker_color = QColor(0, 200, 255)

        # In marker — at the explicit in_pt, or at frame 0 as the implicit
        # range start when only Out is set (so the highlighted region is
        # bookended by visible markers on both sides).
        effective_in = in_pt if in_pt is not None else 0
        in_px = self._frame_to_pixel(effective_in) - self._scroll_offset
        if in_pt is not None and in_px > 0:
            painter.fillRect(0, 0, in_px, total_h, dim_color)
        if 0 <= in_px <= viewport_width:
            painter.setPen(QPen(marker_color, 2))
            painter.drawLine(in_px, 0, in_px, total_h)

        if out_pt is not None:
            out_px = self._frame_to_pixel(out_pt + 1) - self._scroll_offset
            if out_px < viewport_width:
                painter.fillRect(out_px, 0, viewport_width - out_px, total_h, dim_color)
            if 0 <= out_px <= viewport_width:
                painter.setPen(QPen(marker_color, 2))
                painter.drawLine(out_px, 0, out_px, total_h)

    def _paint_crop_strips(self, painter: QPainter, clip_y: int,
                           viewport_width: int):
        """Paint each crop's export segments as blocks on the crop's
        single row, in a band below the group-chip strip. Each block's
        width = the 81-frame export window's source-frame width at the
        current zoom; left edge = the segment's anchor. Color = the
        crop's group color (grey if untagged); orange if the clip is too
        short to host the window; dimmed when the crop or segment is
        inactive. The selected segment gets a 1-px white halo."""
        groups = self._model.groups
        crop_band_y = (clip_y + self._clip_height
                       + self.GROUP_STRIP_RESERVE)
        row_h = self.CROP_STRIP_HEIGHT + self.CROP_STRIP_GAP
        cumulative = 0
        for clip in self._model.clips:
            clip_start_frame = cumulative
            cumulative += clip.duration_frames
            if clip.is_gap or not clip.crop_regions:
                continue
            source = self._sources_ref.get(clip.source_id) \
                if self._sources_ref else None
            if source is None or source.fps <= 0:
                continue
            req = required_source_frames(source.fps)
            clip_src_frames = clip.source_out - clip.source_in + 1
            too_short = req > clip_src_frames
            for i, cr in enumerate(clip.crop_regions):
                strip_y = crop_band_y + i * row_h
                # Base color for the crop (group color / grey / orange).
                if too_short:
                    base_color = QColor(CROP_WARNING_COLOR)
                elif cr.group_id and cr.group_id in groups:
                    base_color = QColor(groups[cr.group_id].color)
                    if not base_color.isValid():
                        base_color = QColor("#888888")
                else:
                    base_color = QColor("#888888")
                # Each segment of the crop is a block on this one row.
                # Collect unclipped pixel spans for every segment so the
                # overlap hatch can be computed (off-screen ones included).
                seg_spans = []
                # Pixel edges of the selected segment — the overlap painter
                # skips its own boundary line there since the selection halo
                # already draws a white border (avoids a doubled-up edge).
                sel_edges = None
                for seg in cr.segments:
                    # clamp_anchor pins an out-of-range anchor to the
                    # clip's rightmost legal frame so the block never
                    # overflows the clip body.
                    anchor_clamped = clamp_anchor(
                        seg.anchor_frame, clip, source.fps)
                    anchor_tl = (clip_start_frame
                                 + (anchor_clamped - clip.source_in))
                    end_tl = anchor_tl + req
                    anchor_px = (self._frame_to_pixel(anchor_tl)
                                 - self._scroll_offset)
                    end_px = (self._frame_to_pixel(end_tl)
                              - self._scroll_offset)
                    width = max(MIN_CROP_STRIP_PX, end_px - anchor_px)
                    seg_spans.append((anchor_px, anchor_px + width))
                    if seg.id == self._selected_segment_id:
                        sel_edges = (anchor_px, anchor_px + width)
                    # Off-screen culling (block paint only — span already
                    # recorded for overlap detection).
                    if anchor_px + width < 0 or anchor_px > viewport_width:
                        continue
                    color = QColor(base_color)
                    # A segment dims if its crop OR the segment itself is
                    # inactive (either excludes it from export).
                    color.setAlpha(255 if (cr.active and seg.active) else 76)
                    draw_x = max(0, anchor_px)
                    draw_w = min(viewport_width, anchor_px + width) - draw_x
                    if draw_w <= 0:
                        continue
                    if seg.id == self._selected_segment_id:
                        halo = QColor(255, 255, 255, 220)
                        painter.fillRect(int(draw_x) - 1, int(strip_y) - 1,
                                         int(draw_w) + 2,
                                         int(self.CROP_STRIP_HEIGHT) + 2, halo)
                    painter.fillRect(int(draw_x), int(strip_y),
                                     int(draw_w),
                                     int(self.CROP_STRIP_HEIGHT), color)
                    # Keyframe diamonds for keys inside THIS segment's
                    # window (animation is shared crop-wide; each block
                    # shows the slice it will export).
                    if cr.is_animated():
                        kf_frames = self._collect_kf_frames(
                            cr, anchor_clamped, anchor_clamped + req)
                        if kf_frames:
                            self._paint_kf_diamonds(
                                painter, kf_frames, clip, clip_start_frame,
                                strip_y)
                # Hatch the spans where two+ of this crop's segments overlap.
                self._paint_segment_overlaps(
                    painter, seg_spans, strip_y, viewport_width, sel_edges)

    def _paint_segment_overlaps(self, painter: QPainter, spans,
                                strip_y: int, viewport_width: int,
                                selected_edges=None):
        """Hatch the pixel spans where two or more of a crop's segments
        overlap, so a shared export window reads as striped rather than
        one block silently sitting on top of another. Each overlap's
        edges (where one segment ends) get a light vertical line so the
        user can tell where the shared region stops and the part that
        doesn't belong to the selected segment begins.

        ``selected_edges`` is the (left, right) pixel pair of the selected
        segment, if any. The selection halo already paints a white border
        there, so we skip our own line at those x's — otherwise halo +
        line stack into a doubled-width edge."""
        if len(spans) < 2:
            return
        hatch = QBrush(QColor(255, 255, 255, 150),
                       Qt.BrushStyle.FDiagPattern)
        # Edge lines are drawn as 1px filled columns (NOT a QPen stroke) at
        # the same alpha as the selection halo, so a boundary marked by the
        # halo and one marked here render pixel-identically (a width-1 pen
        # stroke rasterises softer/thinner than a fillRect column).
        edge_color = QColor(255, 255, 255, 220)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        h = int(self.CROP_STRIP_HEIGHT)
        y0 = int(strip_y)
        # Edge columns span the selected-segment halo's extent (1px above /
        # below the strip) so both boundaries read the same height whether
        # or not one of them sits on the selected (haloed) segment's edge.
        line_y0 = y0 - 1
        line_h = h + 2

        def _is_selected_edge(edge: float) -> bool:
            if selected_edges is None:
                return False
            return any(abs(edge - e) < 0.5 for e in selected_edges)

        n = len(spans)
        for i in range(n):
            for j in range(i + 1, n):
                lo = max(spans[i][0], spans[j][0])
                hi = min(spans[i][1], spans[j][1])
                if hi <= lo:
                    continue
                dx = max(0.0, lo)
                dw = min(viewport_width, hi) - dx
                if dw <= 0:
                    continue
                painter.fillRect(int(dx), y0, int(dw), h, hatch)
                # 1px filled column at each overlap boundary (= a segment
                # edge). Skip edges already drawn by the selection halo,
                # and only draw on-screen.
                for edge in (lo, hi):
                    if _is_selected_edge(edge):
                        continue
                    if 0 <= edge <= viewport_width:
                        painter.fillRect(int(edge), line_y0, 1, line_h,
                                         edge_color)
        painter.restore()

    @staticmethod
    def _collect_kf_frames(cr, lo: int, hi: int):
        """Unique source frames in [lo, hi) that have at least one key
        across the four axis tracks."""
        out = set()
        for track in (cr.x_track, cr.y_track, cr.w_track, cr.h_track):
            for k in track.keys:
                if lo <= k.source_frame < hi:
                    out.add(k.source_frame)
        return sorted(out)

    def _paint_kf_diamonds(self, painter: QPainter, kf_frames,
                           clip, clip_start_frame: int, strip_y: int):
        """Paint a small diamond per unique keyframe source frame.
        Diamonds sit centered vertically on the strip and clamp
        horizontally to the visible viewport."""
        size = max(3, int(self.CROP_STRIP_HEIGHT * 0.9))
        half = size / 2.0
        cy = strip_y + self.CROP_STRIP_HEIGHT / 2.0
        diamond_color = QColor("#ffffff")
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#202020"), 1))
        painter.setBrush(QBrush(diamond_color))
        viewport_w = self.width()
        for kf in kf_frames:
            timeline_frame = clip_start_frame + (kf - clip.source_in)
            cx = self._frame_to_pixel(timeline_frame) - self._scroll_offset
            if cx + half < 0 or cx - half > viewport_w:
                continue
            poly = QPolygon([
                QPoint(int(cx),         int(cy - half)),
                QPoint(int(cx + half),  int(cy)),
                QPoint(int(cx),         int(cy + half)),
                QPoint(int(cx - half),  int(cy)),
            ])
            painter.drawPolygon(poly)
        painter.restore()

    def _hit_crop_strip(self, x: float, y: float):
        """Return ``(clip_id, crop_id, segment_id, original_source_anchor,
        clip_timeline_start, req_src_frames)`` if the point hits a crop
        segment block, else None. Iterates clips / crops / segments in
        reverse so the topmost overlapping block wins."""
        clip_y = self.HEADER_HEIGHT + ui_scale().px(2)
        crop_band_y = clip_y + self._clip_height + self.GROUP_STRIP_RESERVE
        band_h = self._crop_band_height()
        if band_h <= 0 or y < crop_band_y or y >= crop_band_y + band_h:
            return None
        row_h = self.CROP_STRIP_HEIGHT + self.CROP_STRIP_GAP
        # Build (clip, start_frame) pairs once, then iterate reversed.
        starts = []
        cumulative = 0
        for clip in self._model.clips:
            starts.append((clip, cumulative))
            cumulative += clip.duration_frames
        for clip, clip_start_frame in reversed(starts):
            if clip.is_gap or not clip.crop_regions:
                continue
            source = self._sources_ref.get(clip.source_id) \
                if self._sources_ref else None
            if source is None or source.fps <= 0:
                continue
            req = required_source_frames(source.fps)
            for i in range(len(clip.crop_regions) - 1, -1, -1):
                cr = clip.crop_regions[i]
                strip_y = crop_band_y + i * row_h
                if not (strip_y <= y < strip_y + self.CROP_STRIP_HEIGHT):
                    continue
                # Reverse so a later (top-painted) segment wins on overlap.
                for seg in reversed(cr.segments):
                    anchor_clamped = clamp_anchor(
                        seg.anchor_frame, clip, source.fps)
                    anchor_tl = (clip_start_frame
                                 + (anchor_clamped - clip.source_in))
                    end_tl = anchor_tl + req
                    anchor_px = (self._frame_to_pixel(anchor_tl)
                                 - self._scroll_offset)
                    end_px = (self._frame_to_pixel(end_tl)
                              - self._scroll_offset)
                    width = max(MIN_CROP_STRIP_PX, end_px - anchor_px)
                    if anchor_px <= x < anchor_px + width:
                        return (clip.id, cr.id, seg.id, seg.anchor_frame,
                                clip_start_frame, req)
        return None

    def _paint_playhead(self, painter: QPainter, clip_y: int, total_height: int):
        px = self._frame_to_pixel(self._playhead_frame) - self._scroll_offset
        if 0 <= px <= self.width():
            painter.setPen(QPen(PLAYHEAD_COLOR, 2))
            painter.drawLine(px, 0, px, total_height)
            # Playhead handle (triangle at top)
            _w = ui_scale().px(6)
            _h = ui_scale().px(8)
            painter.setBrush(QBrush(PLAYHEAD_COLOR))
            painter.drawPolygon([
                QPoint(px - _w, 0),
                QPoint(px + _w, 0),
                QPoint(px, _h),
            ])

    # --- Mouse interaction ---

    def _track_bottom_y(self) -> int:
        # Bottom of the clip body + group-label strip + crop-strip band.
        # This is where the resize handle sits and where the marquee area
        # starts.
        return (self.HEADER_HEIGHT + ui_scale().px(2)
                + self._clip_height + self.GROUP_STRIP_RESERVE
                + self._crop_band_height())

    def mousePressEvent(self, event: QMouseEvent):
        # Quick-cut: right-click while scrubbing (drag or scrub-follow mode)
        if event.button() == Qt.MouseButton.RightButton:
            if self._dragging_playhead or self._scrub_follow:
                self.cut_requested.emit(self._playhead_frame)
                return

        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle-click: start panning
            self._panning = True
            self._pan_start_x = int(event.position().x())
            self._pan_start_offset = self._scroll_offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.pan_started.emit()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()

            # Crop-strip hit-test runs BEFORE the ruler / clip / marquee
            # branches so dragging a strip doesn't move the playhead or
            # start a marquee. Selecting on press means a no-drag click
            # still routes selection to the panel + preview overlay.
            strip_hit = self._hit_crop_strip(x, y)
            if strip_hit is not None:
                clip_id, crop_id, seg_id, orig, start_tl, req = strip_hit
                alt = bool(event.modifiers()
                           & Qt.KeyboardModifier.AltModifier)
                self._crop_strip_clone = False
                if alt:
                    # Alt-drag clones: append a temp segment to drag now,
                    # commit it as an add (one undo) on release.
                    clip = self._model.get_clip_by_id(clip_id)
                    cr = (self._model._find_crop(clip_id, crop_id)
                          if clip is not None else None)
                    src_seg = cr.find_segment(seg_id) if cr is not None else None
                    if cr is not None and src_seg is not None:
                        from core.crop_region import Segment
                        clone = Segment(anchor_frame=src_seg.anchor_frame,
                                        active=src_seg.active)
                        cr.segments = list(cr.segments) + [clone]
                        seg_id = clone.id
                        orig = clone.anchor_frame
                        self._crop_strip_clone = True
                # Was this block already the selected segment? If so, a
                # pure click (no drag) deselects it on release; a drag
                # still moves it. Clones are always freshly selected.
                self._crop_strip_was_selected = (
                    (not self._crop_strip_clone)
                    and seg_id == self._selected_segment_id)
                self._selected_segment_id = seg_id
                self._dragging_crop_strip = (
                    clip_id, crop_id, seg_id, orig, start_tl, req)
                self._crop_strip_press_x = float(x)
                # Select on press for immediate feedback, except when
                # re-pressing the already-selected block (defer to release
                # so a drag isn't pre-empted by a deselect).
                if not self._crop_strip_was_selected:
                    self.crop_clicked.emit(clip_id, crop_id, seg_id)
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.update()
                return

            # Check if dragging the track bottom edge to resize
            bottom = self._track_bottom_y()
            if abs(y - bottom) <= self.RESIZE_HANDLE_HEIGHT:
                self._resizing_track = True
                self._resize_start_y = int(y)
                self._resize_start_height = self._clip_height
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                return

            # Below track area: start marquee selection
            if y > bottom + self.RESIZE_HANDLE_HEIGHT:
                self._marquee_active = True
                self._marquee_start = QPoint(int(x), int(y))
                self._marquee_end = QPoint(int(x), int(y))
                self._marquee_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                return

            # Click in ruler area = set playhead (works in both modes)
            if y < self.HEADER_HEIGHT:
                self._dragging_playhead = True
                self.scrub_started.emit()
                frame = self._pixel_to_frame(x)
                self.set_playhead(frame)
                return

            # Cut mode: click to split
            if self._edit_mode == EditMode.CUT:
                frame = self._pixel_to_frame(x)
                result = self._model.get_clip_at_position(frame)
                if result and not result[0].is_gap:
                    self.cut_requested.emit(frame)
                return

            # Click in clip area
            frame = self._pixel_to_frame(x)
            result = self._model.get_clip_at_position(frame)
            if result:
                clip, _ = result
                self.clip_clicked.emit(clip.id, event)
            else:
                # Clicked on empty space — set playhead, clear selection
                self._dragging_playhead = True
                self.scrub_started.emit()
                self.set_playhead(frame)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_crop_strip is not None:
            self._do_crop_strip_drag(event.position().x())
            return

        if self._marquee_active:
            self._marquee_end = QPoint(int(event.position().x()), int(event.position().y()))
            self.update()
            return

        if self._resizing_track:
            dy = int(event.position().y()) - self._resize_start_y
            new_h = max(CLIP_HEIGHT_MIN, min(CLIP_HEIGHT_MAX, self._resize_start_height + dy))
            if new_h != self._clip_height:
                self._clip_height = new_h
                self.setMinimumHeight(
                    self._clip_height + self.HEADER_HEIGHT + ui_scale().px(4))
                self.update()
            return

        if self._panning:
            dx = int(event.position().x()) - self._pan_start_x
            self._scroll_offset = max(0, self._pan_start_offset - dx)
            self.update()
            # Sync scrollbar without triggering playhead handlers
            self.scroll_changed.emit()
            return

        if self._dragging_playhead or self._scrub_follow:
            x = event.position().x()
            frame = self._pixel_to_frame(x)
            self.set_playhead(frame)
            if not self._scrub_follow:
                return
            # In scrub-follow, fall through to hover behavior (cursor updates)

        # Hover behavior
        x = event.position().x()
        y = event.position().y()
        bottom = self._track_bottom_y()

        if abs(y - bottom) <= self.RESIZE_HANDLE_HEIGHT:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            self._cut_preview_x = None
        elif self._edit_mode == EditMode.CUT:
            if self.HEADER_HEIGHT <= y <= bottom:
                frame = self._pixel_to_frame(x)
                result = self._model.get_clip_at_position(frame)
                if result and not result[0].is_gap:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                    self._cut_preview_x = int(x)
                    # Only seek when the frame changes (throttle rapid mouse moves)
                    if frame != self._last_preview_frame:
                        self._last_preview_frame = frame
                        self.preview_frame_requested.emit(frame)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    self._cut_preview_x = None
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self._cut_preview_x = None
            self.update()
        else:
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self._cut_preview_x = None

    def leaveEvent(self, event):
        if self._edit_mode == EditMode.CUT and self._cut_preview_x is not None:
            self._cut_preview_x = None
            self.update()
            self.preview_frame_requested.emit(self._playhead_frame)
        super().leaveEvent(event)

    def _do_crop_strip_drag(self, x: float):
        """Live mid-drag mutation. We bypass ``TimelineModel.update_crop_region``
        (which pushes undo each call) and mutate the live ``CropRegion`` in
        place — only the strip needs the visual refresh until release.

        A small dead-zone (3 px) around the press position lets a pure
        click pass through without nudging the anchor by ±1 source frame
        on subpixel mouse jitter."""
        info = self._dragging_crop_strip
        if info is None:
            return
        if abs(x - self._crop_strip_press_x) < 3.0:
            return
        clip_id, crop_id, segment_id, original_anchor, clip_start_tl, req = info
        clip = self._model.get_clip_by_id(clip_id)
        if clip is None or clip.is_gap:
            return
        cr = self._model._find_crop(clip_id, crop_id)
        seg = cr.find_segment(segment_id) if cr is not None else None
        if seg is None:
            return
        # Mouse delta in pixels → source-frame delta, applied to the
        # PRE-DRAG anchor so the block translates by exactly the drag
        # distance regardless of subpixel rounding history.
        dx_px = x - self._crop_strip_press_x
        if self._pixels_per_frame <= 0:
            return
        delta_frames = int(round(dx_px / self._pixels_per_frame))
        source_anchor = original_anchor + delta_frames
        # Clamp to [clip.source_in, clip.source_out - req + 1].
        lo = clip.source_in
        hi = max(lo, clip.source_out - req + 1)
        source_anchor = max(lo, min(hi, source_anchor))
        if source_anchor != seg.anchor_frame:
            seg.anchor_frame = source_anchor
            self.update()

    def _finish_crop_strip_drag(self):
        """Commit the drag as one undo entry. Normal drag → move the
        segment's anchor; Alt-clone → drop the temp clone and re-add it
        through the model at the final position."""
        info = self._dragging_crop_strip
        is_clone = self._crop_strip_clone
        was_selected = self._crop_strip_was_selected
        self._dragging_crop_strip = None
        self._crop_strip_clone = False
        self._crop_strip_was_selected = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if info is None:
            return
        clip_id, crop_id, segment_id, original_anchor, _start_tl, _req = info
        cr = self._model._find_crop(clip_id, crop_id)
        seg = cr.find_segment(segment_id) if cr is not None else None
        if cr is None or seg is None:
            return
        final_anchor = seg.anchor_frame
        seg_active = seg.active
        if is_clone:
            # Remove the temp clone we appended at press, then add it
            # properly through the model (single undo entry).
            cr.segments = [s for s in cr.segments if s.id != segment_id]
            if final_anchor == original_anchor:
                # Pure Alt-click, no drag → don't create a stray clone.
                self.update()
                return
            new_id = self._model.add_crop_segment(
                clip_id, crop_id, final_anchor, active=seg_active)
            if new_id:
                self._selected_segment_id = new_id
                # Re-route selection so the panel highlights the committed
                # clone (the press emitted the now-discarded temp id).
                self.crop_clicked.emit(clip_id, crop_id, new_id)
            return
        if final_anchor == original_anchor:
            # Pure click (no drag). Re-clicking the already-selected
            # segment deselects it (crop stays selected); otherwise it was
            # already selected on press — nothing more to do.
            if was_selected:
                self._selected_segment_id = ""
                self.crop_clicked.emit(clip_id, crop_id, "")
                self.update()
            return
        seg.anchor_frame = original_anchor    # revert; model re-applies
        self._model.move_crop_segment(
            clip_id, crop_id, segment_id, final_anchor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_crop_strip is not None and \
                event.button() == Qt.MouseButton.LeftButton:
            self._finish_crop_strip_drag()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            was_panning = self._panning
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if was_panning:
                self.pan_ended.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._marquee_active:
                self._finish_marquee_selection()
                return
            if self._resizing_track:
                self._resizing_track = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._dragging_playhead:
                self._dragging_playhead = False
                self.scrub_ended.emit()

    def _finish_marquee_selection(self):
        """Compute which clips overlap the marquee and select them."""
        x1 = min(self._marquee_start.x(), self._marquee_end.x())
        x2 = max(self._marquee_start.x(), self._marquee_end.x())

        # Convert pixel range to frame range
        frame_start = self._pixel_to_frame(x1)
        frame_end = self._pixel_to_frame(x2)

        # Walk clips and find overlapping ones
        selected = set() if not self._marquee_ctrl else set(self._model.selected_ids)
        pos = 0
        for clip in self._model.clips:
            clip_end = pos + clip.duration_frames - 1
            if not clip.is_gap and clip_end >= frame_start and pos <= frame_end:
                selected.add(clip.id)
            pos += clip.duration_frames

        self._model.set_selection(selected)

        self._marquee_active = False
        self._marquee_start = None
        self._marquee_end = None
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom
            delta = event.angleDelta().y()
            mouse_x = event.position().x()
            # Frame under mouse before zoom
            frame_under_mouse = (mouse_x + self._scroll_offset - self.TIMELINE_H_PADDING) / self._pixels_per_frame

            factor = 1.15 if delta > 0 else 1 / 1.15
            self._pixels_per_frame = max(0.001, min(20.0, self._pixels_per_frame * factor))

            # Adjust scroll so the frame under mouse stays in place
            new_px = self.TIMELINE_H_PADDING + frame_under_mouse * self._pixels_per_frame
            self._scroll_offset = max(0, int(new_px - mouse_x))
            self.update()
            # Notify parent to update scrollbar
            self.playhead_moved.emit(self._playhead_frame)
            self.scroll_changed.emit()
        else:
            # Horizontal scroll
            delta = event.angleDelta().y()
            self._scroll_offset = max(0, self._scroll_offset - delta)
            self.update()
            self.scroll_changed.emit()

    def resizeEvent(self, event):
        # Widget width changes alter which clips are in the viewport, so
        # nudge the cache to re-examine visibility.
        super().resizeEvent(event)
        self.scroll_changed.emit()

    def _select_adjacent_clip(self, direction: int):
        """Select next (direction=1) or previous (direction=-1) clip and move playhead."""
        clips = self._model.clips
        if not clips:
            return
        # Find current clip at playhead
        result = self._model.get_clip_at_position(self._playhead_frame)
        if result is None:
            # Past the end — select last clip
            idx = len(clips) - 1 if direction < 0 else 0
        else:
            current_clip, _ = result
            idx = self._model.get_clip_index(current_clip.id)
            idx += direction
        idx = max(0, min(idx, len(clips) - 1))
        target = clips[idx]
        self._model.select_clip(target.id)
        start = self._model.get_clip_timeline_start(target.id)
        self.set_playhead(start)
        self.ensure_playhead_visible()

    # --- Public timeline-action methods ---
    #
    # These were previously inlined in keyPressEvent. They're now exposed as
    # methods so MainWindow can wire customizable QShortcuts to them via the
    # ShortcutManager. The shortcuts (Left/Right/Home/End/Up/Down by default)
    # are owned by core.shortcuts.SHORTCUTS — change the bindings in the
    # Keyboard Shortcuts dialog.

    def step_playhead(self, frames: int):
        """Move the playhead by ``frames`` (negative for backward)."""
        self.set_playhead(self._playhead_frame + frames)
        self.ensure_playhead_visible()

    def go_to_start(self):
        self.set_playhead(0)
        self.set_scroll_offset(0)

    def go_to_end(self):
        total = self._model.get_total_duration_frames()
        self.set_playhead(total - 1 if total > 0 else 0)
        self.ensure_playhead_visible()

    def select_adjacent_clip(self, direction: int):
        """Select the next (1) or previous (-1) clip on the timeline."""
        self._select_adjacent_clip(direction)

    def keyPressEvent(self, event: QKeyEvent):
        # All formerly-hardcoded keys (arrow stepping, Home/End, Up/Down clip
        # navigation) are now QShortcut-driven; nothing left to handle here.
        super().keyPressEvent(event)


class TimelineWidget(QWidget):
    """Timeline widget with strip + horizontal scrollbar."""

    playhead_changed = Signal(int)
    scrub_started = Signal()
    scrub_ended = Signal()
    pan_started = Signal()
    pan_ended = Signal()
    clip_clicked = Signal(str, object)
    preview_frame_requested = Signal(int)
    cut_requested = Signal(int)
    thumbnails_toggled = Signal(bool)
    hq_thumbnails_toggled = Signal(bool)
    cache_thumbnails_clicked = Signal()  # button hit; main_window decides start vs cancel
    sources_dropped = Signal(list, int)  # source_ids, insert_frame
    files_dropped = Signal(list, int)    # file paths, insert_frame
    crop_clicked = Signal(str, str, str)  # clip_id, crop_id, segment_id (block hit)

    def __init__(self, model: TimelineModel, parent=None):
        super().__init__(parent)
        self._model = model

        self._strip = TimelineStrip(model)
        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setMinimum(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._strip, 1)
        layout.addWidget(self._scrollbar)

        _us = ui_scale()
        # Thumbnail toggle — small checkable overlay in the strip's top-right.
        self._thumb_toggle = QToolButton(self._strip)
        self._thumb_toggle.setIcon(icon("thumbnails"))
        self._thumb_toggle.setIconSize(QSize(_us.px(14), _us.px(14)))
        self._thumb_toggle.setCheckable(True)
        self._thumb_toggle.setChecked(True)
        self._thumb_toggle.setToolTip("Toggle clip thumbnails")
        self._thumb_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thumb_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumb_toggle.setFixedSize(_us.px(22), _us.px(22))
        self._thumb_toggle.setStyleSheet(
            "QToolButton {"
            " background-color: rgba(58, 58, 58, 220);"
            " border: 1px solid #555;"
            " border-radius: 3px;"
            "}"
            "QToolButton:hover { background-color: rgba(85, 85, 85, 230); }"
            "QToolButton:checked {"
            " background-color: #5577aa;"
            " border-color: #6688bb;"
            "}"
        )
        self._thumb_toggle.toggled.connect(self._on_thumb_toggled)
        self._thumb_toggle.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._thumb_toggle.customContextMenuRequested.connect(self._show_thumb_menu)

        # Cache-thumbnails button — left of the toggle. Click runs a bulk
        # disk-cache pass for clip-boundary frames; click again while
        # running cancels.
        self._cache_thumb_btn = QToolButton(self._strip)
        self._cache_thumb_btn.setIcon(icon("save"))
        self._cache_thumb_btn.setIconSize(QSize(_us.px(14), _us.px(14)))
        self._cache_thumb_btn.setToolTip(
            "Cache clip-boundary thumbnails to disk\n"
            "Respects in/out range if set")
        self._cache_thumb_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cache_thumb_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._cache_thumb_btn.setFixedSize(_us.px(22), _us.px(22))
        self._cache_thumb_btn.setStyleSheet(
            "QToolButton {"
            " background-color: rgba(58, 58, 58, 220);"
            " border: 1px solid #555;"
            " border-radius: 3px;"
            "}"
            "QToolButton:hover { background-color: rgba(85, 85, 85, 230); }"
        )
        self._cache_thumb_btn.clicked.connect(self.cache_thumbnails_clicked.emit)

        self._strip.installEventFilter(self)
        self._position_thumb_toggle()
        ui_scale().changed.connect(self._on_ui_scale_changed)

        # HQ generation is part of the advanced options — default on.
        self._hq_thumbnails_enabled = True

        self._strip.playhead_moved.connect(self._on_playhead_moved)
        self._strip.scrub_started.connect(self.scrub_started.emit)
        self._strip.scrub_ended.connect(self.scrub_ended.emit)
        self._strip.pan_started.connect(self.pan_started.emit)
        self._strip.pan_ended.connect(self.pan_ended.emit)
        self._strip.scroll_changed.connect(self._update_scrollbar)
        self._strip.clip_clicked.connect(self.clip_clicked.emit)
        self._strip.preview_frame_requested.connect(self.preview_frame_requested.emit)
        self._strip.cut_requested.connect(self.cut_requested.emit)
        self._strip.sources_dropped.connect(self.sources_dropped.emit)
        self._strip.files_dropped.connect(self.files_dropped.emit)
        self._strip.crop_clicked.connect(self.crop_clicked.emit)
        self._scrollbar.valueChanged.connect(self._strip.set_scroll_offset)
        self._model.clips_changed.connect(self._update_scrollbar)

    @property
    def strip(self) -> TimelineStrip:
        return self._strip

    def set_sources_ref(self, sources: dict):
        """Forward MainWindow's source dict to the strip so drag previews
        can render per-source thumbnails."""
        self._strip.set_sources_ref(sources)

    def set_selected_crop_id(self, crop_id: str):
        """External crop-selection sync (panel ⇄ preview ⇄ timeline)."""
        self._strip.set_selected_crop_id(crop_id)

    def set_selected_segment_id(self, segment_id: str):
        """External segment-selection sync (panel ⇄ timeline)."""
        self._strip.set_selected_segment_id(segment_id)

    @property
    def thumbnails_enabled(self) -> bool:
        return self._thumb_toggle.isChecked()

    @property
    def hq_thumbnails_enabled(self) -> bool:
        return self._hq_thumbnails_enabled

    def _on_thumb_toggled(self, checked: bool):
        self._strip.set_thumbnails_enabled(checked)
        self._update_thumb_icon()
        self.thumbnails_toggled.emit(checked)

    def _show_thumb_menu(self, pos):
        menu = QMenu(self._thumb_toggle)
        hq_action = QAction("Generate HQ Thumbnails", menu)
        hq_action.setCheckable(True)
        hq_action.setChecked(self._hq_thumbnails_enabled)
        hq_action.toggled.connect(self._on_hq_thumb_toggled)
        menu.addAction(hq_action)
        menu.exec(self._thumb_toggle.mapToGlobal(pos))

    def _on_hq_thumb_toggled(self, enabled: bool):
        if self._hq_thumbnails_enabled == enabled:
            return
        self._hq_thumbnails_enabled = enabled
        self._update_thumb_icon()
        self.hq_thumbnails_toggled.emit(enabled)

    def _update_thumb_icon(self):
        # Degraded = master on but HQ off → LQ placeholders only.
        degraded = self._thumb_toggle.isChecked() and not self._hq_thumbnails_enabled
        self._thumb_toggle.setIcon(
            icon("thumbnails-degraded") if degraded else icon("thumbnails"))

    def _position_thumb_toggle(self):
        s = ui_scale()
        margin = s.px(6)
        gap = s.px(4)
        x = self._strip.width() - self._thumb_toggle.width() - margin
        y = margin
        self._thumb_toggle.move(x, y)
        self._thumb_toggle.raise_()
        cx = x - gap - self._cache_thumb_btn.width()
        self._cache_thumb_btn.move(cx, y)
        self._cache_thumb_btn.raise_()

    def _on_ui_scale_changed(self):
        s = ui_scale()
        for btn in (self._thumb_toggle, self._cache_thumb_btn):
            btn.setIconSize(QSize(s.px(14), s.px(14)))
            btn.setFixedSize(s.px(22), s.px(22))
        self._position_thumb_toggle()

    def eventFilter(self, obj, event):
        if obj is self._strip and event.type() == event.Type.Resize:
            self._position_thumb_toggle()
        return super().eventFilter(obj, event)

    def set_playhead(self, frame: int):
        self._strip.set_playhead(frame)

    def set_fps(self, fps: float):
        self._strip.set_fps(fps)

    def _on_playhead_moved(self, frame: int):
        self._update_scrollbar()
        self.playhead_changed.emit(frame)

    def _update_scrollbar(self):
        total_w = self._strip.get_total_width()
        visible_w = self._strip.width()
        self._scrollbar.setMaximum(max(0, total_w - visible_w))
        self._scrollbar.setPageStep(visible_w)
        # Sync scrollbar position to strip's scroll offset without triggering feedback
        self._scrollbar.blockSignals(True)
        self._scrollbar.setValue(self._strip._scroll_offset)
        self._scrollbar.blockSignals(False)
