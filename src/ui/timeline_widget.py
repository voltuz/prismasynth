import logging
from enum import IntEnum
from typing import Optional, List

from PySide6.QtWidgets import QWidget, QScrollBar, QVBoxLayout, QToolButton, QMenu
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QAction,
    QMouseEvent, QWheelEvent, QPaintEvent, QKeyEvent,
)

from core.timeline import TimelineModel
from core.clip import Clip
from ui.icon_loader import icon

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

CLIP_HEIGHT_DEFAULT = 60
CLIP_HEIGHT_MIN = 30
CLIP_HEIGHT_MAX = 200
HEADER_HEIGHT = 48
TIMELINE_H_PADDING = 4  # small internal padding for playhead clearance
RESIZE_HANDLE_HEIGHT = 5  # pixels at bottom edge for drag resize
THUMB_HIDE_THRESHOLD = 30  # px clip width — below this, skip thumbnail paint and generation
PLAYHEAD_COLOR = QColor(255, 50, 50)
SELECTION_BORDER = QColor(255, 255, 100)
GAP_COLOR = QColor(30, 30, 30)
BG_COLOR = QColor(40, 40, 40)
RULER_BG = QColor(50, 50, 50)
RULER_TEXT = QColor(180, 180, 180)
THUMBNAIL_WIDTH = 64


class TimelineStrip(QWidget):
    """Custom-painted timeline strip showing clips as colored blocks."""

    playhead_moved = Signal(int)   # timeline frame
    scrub_started = Signal()       # user started dragging playhead
    scrub_ended = Signal()         # user released playhead drag
    scroll_changed = Signal()      # scroll offset changed (no playhead change)
    clip_clicked = Signal(str, object)  # clip_id, QMouseEvent (for modifier keys)
    preview_frame_requested = Signal(int)  # cut-mode hover scrub (no playhead move)
    cut_requested = Signal(int)    # cut-mode click at timeline frame

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

        self.setMinimumHeight(CLIP_HEIGHT_MIN + HEADER_HEIGHT + 4)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._model.clips_changed.connect(self.update)
        self._model.selection_changed.connect(self.update)
        self._model.in_out_changed.connect(self.update)

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

    def set_thumbnail(self, clip_id: str, position: str, pixmap: QPixmap):
        self._thumbnails[(clip_id, position)] = pixmap
        self.update()

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
        clip_y = HEADER_HEIGHT + 2
        self._paint_clips(painter, clip_y, w)

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

        painter.end()

    def _paint_ruler(self, painter: QPainter, width: int):
        painter.fillRect(0, 0, width, HEADER_HEIGHT, RULER_BG)
        painter.setPen(QPen(RULER_TEXT))
        font = QFont("Segoe UI", 8)
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

        start_sec = max(0, self._scroll_offset - TIMELINE_H_PADDING) / (self._pixels_per_frame * self._fps)
        end_sec = max(0, self._scroll_offset + width - TIMELINE_H_PADDING) / (self._pixels_per_frame * self._fps)
        t = (int(start_sec / interval)) * interval
        while t <= end_sec:
            x = self._frame_to_pixel(int(t * self._fps)) - self._scroll_offset
            if 0 <= x <= width:
                painter.drawLine(x, HEADER_HEIGHT - 6, x, HEADER_HEIGHT)
                ti = int(t)
                if interval >= 3600:
                    label = f"{ti // 3600}:{(ti % 3600) // 60:02d}:{ti % 60:02d}"
                else:
                    label = f"{ti // 60}:{ti % 60:02d}"
                painter.drawText(x + 3, HEADER_HEIGHT - 8, label)
            t += interval

    def _frame_to_pixel(self, frame: int) -> int:
        """Convert a timeline frame number to pixel position. Single source of truth."""
        return TIMELINE_H_PADDING + int(frame * self._pixels_per_frame)

    def _pixel_to_frame(self, screen_x: float) -> int:
        """Convert a screen x coordinate to timeline frame number.
        Clamps to content area so the playhead can't enter the padding."""
        # Clamp screen_x to content bounds
        x = max(TIMELINE_H_PADDING, screen_x)
        total = self._model.get_total_duration_frames()
        if total > 0:
            max_px = self._frame_to_pixel(total - 1) - self._scroll_offset
            x = min(x, max_px)
        return max(0, int((x + self._scroll_offset - TIMELINE_H_PADDING)
                          / self._pixels_per_frame))

    def get_visible_clip_ids(self):
        """Return IDs of non-gap clips currently visible in the viewport.

        Clips narrower than THUMB_HIDE_THRESHOLD are excluded so the thumbnail
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
            if clip_w < THUMB_HIDE_THRESHOLD:
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
            pad = 6
            th = self._clip_height - pad * 2
            tw = int(th * 16 / 9)

            if self._thumbnails_enabled and clip_w >= THUMB_HIDE_THRESHOLD:
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
            if clip_w > tw * 2 + 40:
                painter.setPen(QPen(QColor(240, 240, 240)))
                font = QFont("Segoe UI", 8)
                painter.setFont(font)
                dur_secs = clip.duration_frames / self._fps if self._fps > 0 else 0
                if dur_secs >= 60:
                    label = f"{int(dur_secs)//60}:{int(dur_secs)%60:02d}"
                else:
                    label = f"{dur_secs:.1f}s"
                text_rect = QRect(screen_x + tw, y, clip_w - tw * 2, self._clip_height)
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

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

        if in_pt is not None:
            in_px = self._frame_to_pixel(in_pt) - self._scroll_offset
            dim_right = max(0, in_px)
            if dim_right > 0:
                painter.fillRect(0, 0, dim_right, total_h, dim_color)
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

    def _paint_playhead(self, painter: QPainter, clip_y: int, total_height: int):
        px = self._frame_to_pixel(self._playhead_frame) - self._scroll_offset
        if 0 <= px <= self.width():
            painter.setPen(QPen(PLAYHEAD_COLOR, 2))
            painter.drawLine(px, 0, px, total_height)
            # Playhead handle (triangle at top)
            painter.setBrush(QBrush(PLAYHEAD_COLOR))
            painter.drawPolygon([
                QPoint(px - 6, 0),
                QPoint(px + 6, 0),
                QPoint(px, 8),
            ])

    # --- Mouse interaction ---

    def _track_bottom_y(self) -> int:
        return HEADER_HEIGHT + 2 + self._clip_height

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
            return

        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()

            # Check if dragging the track bottom edge to resize
            bottom = self._track_bottom_y()
            if abs(y - bottom) <= RESIZE_HANDLE_HEIGHT:
                self._resizing_track = True
                self._resize_start_y = int(y)
                self._resize_start_height = self._clip_height
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                return

            # Below track area: start marquee selection
            if y > bottom + RESIZE_HANDLE_HEIGHT:
                self._marquee_active = True
                self._marquee_start = QPoint(int(x), int(y))
                self._marquee_end = QPoint(int(x), int(y))
                self._marquee_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                return

            # Click in ruler area = set playhead (works in both modes)
            if y < HEADER_HEIGHT:
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
        if self._marquee_active:
            self._marquee_end = QPoint(int(event.position().x()), int(event.position().y()))
            self.update()
            return

        if self._resizing_track:
            dy = int(event.position().y()) - self._resize_start_y
            new_h = max(CLIP_HEIGHT_MIN, min(CLIP_HEIGHT_MAX, self._resize_start_height + dy))
            if new_h != self._clip_height:
                self._clip_height = new_h
                self.setMinimumHeight(self._clip_height + HEADER_HEIGHT + 4)
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

        if abs(y - bottom) <= RESIZE_HANDLE_HEIGHT:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            self._cut_preview_x = None
        elif self._edit_mode == EditMode.CUT:
            if HEADER_HEIGHT <= y <= bottom:
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

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
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
            frame_under_mouse = (mouse_x + self._scroll_offset - TIMELINE_H_PADDING) / self._pixels_per_frame

            factor = 1.15 if delta > 0 else 1 / 1.15
            self._pixels_per_frame = max(0.001, min(20.0, self._pixels_per_frame * factor))

            # Adjust scroll so the frame under mouse stays in place
            new_px = TIMELINE_H_PADDING + frame_under_mouse * self._pixels_per_frame
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

    def keyPressEvent(self, event: QKeyEvent):
        # Arrow keys for frame stepping
        if event.key() == Qt.Key.Key_Left:
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self.set_playhead(self._playhead_frame - step)
            self.ensure_playhead_visible()
        elif event.key() == Qt.Key.Key_Right:
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self.set_playhead(self._playhead_frame + step)
            self.ensure_playhead_visible()
        elif event.key() == Qt.Key.Key_Home:
            self.set_playhead(0)
            self.set_scroll_offset(0)
        elif event.key() == Qt.Key.Key_End:
            total = self._model.get_total_duration_frames()
            self.set_playhead(total - 1 if total > 0 else 0)
            self.ensure_playhead_visible()
        elif event.key() == Qt.Key.Key_Down:
            self._select_adjacent_clip(1)
        elif event.key() == Qt.Key.Key_Up:
            self._select_adjacent_clip(-1)
        else:
            super().keyPressEvent(event)


class TimelineWidget(QWidget):
    """Timeline widget with strip + horizontal scrollbar."""

    playhead_changed = Signal(int)
    scrub_started = Signal()
    scrub_ended = Signal()
    clip_clicked = Signal(str, object)
    preview_frame_requested = Signal(int)
    cut_requested = Signal(int)
    thumbnails_toggled = Signal(bool)
    hq_thumbnails_toggled = Signal(bool)

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

        # Thumbnail toggle — small checkable overlay in the strip's top-right.
        self._thumb_toggle = QToolButton(self._strip)
        self._thumb_toggle.setIcon(icon("thumbnails"))
        self._thumb_toggle.setIconSize(QSize(14, 14))
        self._thumb_toggle.setCheckable(True)
        self._thumb_toggle.setChecked(True)
        self._thumb_toggle.setToolTip("Toggle clip thumbnails")
        self._thumb_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thumb_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumb_toggle.setFixedSize(22, 22)
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
        self._strip.installEventFilter(self)
        self._position_thumb_toggle()

        # HQ generation is part of the advanced options — default on.
        self._hq_thumbnails_enabled = True

        self._strip.playhead_moved.connect(self._on_playhead_moved)
        self._strip.scrub_started.connect(self.scrub_started.emit)
        self._strip.scrub_ended.connect(self.scrub_ended.emit)
        self._strip.scroll_changed.connect(self._update_scrollbar)
        self._strip.clip_clicked.connect(self.clip_clicked.emit)
        self._strip.preview_frame_requested.connect(self.preview_frame_requested.emit)
        self._strip.cut_requested.connect(self.cut_requested.emit)
        self._scrollbar.valueChanged.connect(self._strip.set_scroll_offset)
        self._model.clips_changed.connect(self._update_scrollbar)

    @property
    def strip(self) -> TimelineStrip:
        return self._strip

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
        margin = 6
        x = self._strip.width() - self._thumb_toggle.width() - margin
        y = margin
        self._thumb_toggle.move(x, y)
        self._thumb_toggle.raise_()

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
