"""Transparent overlay that lets the user draw / move / resize per-clip
``CropRegion`` rectangles on top of the mpv preview.

Lives as a child widget of ``PreviewWidget`` and renders via QPainter on a
``WA_TransparentForMouseEvents=False`` widget — so it intercepts mouse
events only when edit mode is on. When edit mode is off, the overlay is
hidden and the preview behaves exactly as before.

All crop geometry is stored in **source-pixel coordinates** on the
``CropRegion``. The overlay maps to widget-pixel coordinates on the fly
using the host PreviewWidget's render geometry + mpv pan state, so zoom /
pan in the preview do not move the underlying crop data.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QPainter, QPen, QPolygon,
)
from PySide6.QtWidgets import QWidget

from core.crop_region import CropRegion, resolve_aspect


# Handle hit-test size in widget pixels (each side of the square handle).
HANDLE_PX = 10
# Minimum crop size in source pixels.
MIN_CROP_SIZE = 8


# Hit-test results for mouse interaction.
HIT_NONE = "none"
HIT_BODY = "body"
HIT_OUTSIDE = "outside"  # for empty area = draw-new
# Handles (8 around the bounding box).
HIT_NW = "nw"
HIT_N = "n"
HIT_NE = "ne"
HIT_E = "e"
HIT_SE = "se"
HIT_S = "s"
HIT_SW = "sw"
HIT_W = "w"

_HANDLES = (HIT_NW, HIT_N, HIT_NE, HIT_E, HIT_SE, HIT_S, HIT_SW, HIT_W)

# Map handle to cursor shape.
_HANDLE_CURSORS = {
    HIT_NW: Qt.CursorShape.SizeFDiagCursor,
    HIT_SE: Qt.CursorShape.SizeFDiagCursor,
    HIT_NE: Qt.CursorShape.SizeBDiagCursor,
    HIT_SW: Qt.CursorShape.SizeBDiagCursor,
    HIT_N:  Qt.CursorShape.SizeVerCursor,
    HIT_S:  Qt.CursorShape.SizeVerCursor,
    HIT_E:  Qt.CursorShape.SizeHorCursor,
    HIT_W:  Qt.CursorShape.SizeHorCursor,
}


class CropOverlay(QWidget):
    """Drawable / editable crop-rectangle overlay for the preview."""

    # Crop selection emitted when the user clicks on a rectangle.
    crop_selected = Signal(str)
    # User changed the crop's geometry (x/y/w/h) — owner should persist.
    crop_geometry_changed = Signal(str, int, int, int, int)
    # User finished drawing a new rectangle — owner should add it.
    new_crop_drawn = Signal(int, int, int, int)
    # Delete pressed while a crop is selected.
    delete_requested = Signal(str)

    def __init__(self, preview, parent=None):
        # Top-level frameless tool window — NOT a child widget.
        #
        # WA_TranslucentBackground on a child widget on Windows implicitly
        # promotes the widget to a detached native top-level window
        # without inheriting its parent's screen position. That caused
        # three reported symptoms: a static grey rectangle over the
        # preview (alpha couldn't composite against mpv's sibling HWND),
        # crops that couldn't reach the visible top-left because the
        # overlay's local coord frame started somewhere off the preview,
        # and a mouse-to-paint offset for the same reason.
        #
        # Making the overlay an *explicit* top-level window with
        # `Qt.Tool` (tied to the parent window for show/hide / z-order)
        # gives proper per-pixel alpha through DWM and lets us pin its
        # screen geometry to the PreviewWidget's screen rect on every
        # resize / move event.
        super().__init__(
            preview,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool,
        )
        self._preview = preview
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                          True)
        self.setVisible(False)
        # We want keyboard input for Delete / Escape only when edit mode is
        # on; we set this dynamically.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Receive mouseMoveEvent on hover (not just during drag) so the
        # cursor logic in mouseMoveEvent → _cursor_for_position can
        # update over handles / body / empty area. Default is False,
        # which is why hover cursors never changed.
        self.setMouseTracking(True)

        self._edit_mode: bool = False
        self._clip_id: Optional[str] = None
        self._source_w: int = 0
        self._source_h: int = 0
        self._crops: List[CropRegion] = []
        # Selected crop id ("" = none selected).
        self._selected_id: str = ""
        # Group color lookup by crop id (set by owner so we don't reach
        # into TimelineModel from here).
        self._group_color: dict[str, str] = {}

        # Drag state
        self._drag_kind: str = ""        # "draw" | "move" | "resize" | ""
        self._drag_crop_id: str = ""
        self._drag_handle: str = ""       # for resize
        self._drag_start_widget: QPoint = QPoint()
        # Source-space starting rect (x, y, w, h) for the dragged crop.
        self._drag_start_rect: tuple = (0, 0, 0, 0)
        # For draw-new: anchor (corner where the drag started) in source coords.
        self._draw_anchor_src: tuple = (0, 0)
        self._draw_aspect: Optional[Tuple[int, int]] = None  # locked ratio

    # ------------------------------------------------------------------
    # Configuration from owner
    # ------------------------------------------------------------------

    def sync_geometry_to_preview(self):
        """Pin the overlay's *screen* geometry to PreviewWidget's screen
        rect. Caller drives this from PreviewWidget's resize / move
        events and from MainWindow's move / state-change events; the
        overlay is a top-level window so it doesn't inherit those
        automatically."""
        if not self._preview.isVisible():
            return
        top_left = self._preview.mapToGlobal(QPoint(0, 0))
        size = self._preview.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        self.setGeometry(top_left.x(), top_left.y(),
                         size.width(), size.height())

    def set_clip_state(self, clip_id: Optional[str], source_w: int,
                       source_h: int,
                       crops: Optional[List[CropRegion]] = None,
                       group_colors: Optional[dict] = None):
        """Tell the overlay which clip's crops to display. Pass
        ``clip_id=None`` (or empty crops on a non-gap clip) to clear.
        Visibility is decided automatically by ``_refresh_visibility``
        — callers do not need to also call show()/hide().

        This is the single entry point for "what to draw": called on
        selection_changed, clips_changed, groups_changed."""
        self._clip_id = clip_id
        self._source_w = int(source_w) if source_w else 0
        self._source_h = int(source_h) if source_h else 0
        self._crops = list(crops or [])
        if group_colors is not None:
            self._group_color = dict(group_colors)
        # Drop stale selection.
        if self._selected_id and not any(
                cr.id == self._selected_id for cr in self._crops):
            self._selected_id = ""
        self._refresh_visibility()
        self.update()

    def set_edit_mode(self, on: bool):
        """Toggle interactivity (handles, mouse / Delete / Esc input,
        and the draw-new affordance). Visibility is independent — active
        crops still render as outlines when edit mode is off."""
        self._edit_mode = bool(on)
        # Hide/re-show around the WA_TransparentForMouseEvents flip.
        # On a top-level Qt.Tool HWND, Windows only re-emits the
        # WS_EX_TRANSPARENT ex-style at show() time — toggling the Qt
        # attribute on a visible native window leaves the OS ex-style
        # stale, so DWM keeps routing clicks straight through to the
        # mpv HWND below and the overlay never receives mousePress.
        # The hide()/show() cycle via _refresh_visibility() forces QPA
        # to re-emit the correct ex-style.
        was_visible = self.isVisible()
        if was_visible:
            self.hide()
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on)
        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus if on else Qt.FocusPolicy.NoFocus)
        if not on:
            # Leaving edit mode: drop selection + any in-flight drag so
            # the next entry into edit mode starts fresh.
            self._selected_id = ""
            self._cancel_drag()
        self._refresh_visibility()
        if on and self.isVisible():
            # Qt.Tool on Windows can swallow the first click for window
            # activation — activateWindow() ensures it lands as a real
            # mousePressEvent.
            self.activateWindow()
            self.setFocus()
        self.update()

    def _refresh_visibility(self):
        """Show iff there's a clip context AND (edit_mode is on OR at
        least one crop on the clip is active). Hide otherwise. Always
        pins screen geometry before showing so the first paint lands
        aligned with PreviewWidget."""
        has_clip = bool(self._clip_id) and self._source_w > 0 and self._source_h > 0
        has_active = has_clip and any(c.active for c in self._crops)
        should_show = has_clip and (self._edit_mode or has_active)
        if should_show:
            if not self.isVisible():
                self.sync_geometry_to_preview()
                self.show()
                self.raise_()
        else:
            if self.isVisible():
                self.hide()

    def set_selected_crop(self, crop_id: str):
        if not crop_id:
            crop_id = ""
        if crop_id == self._selected_id:
            return
        self._selected_id = crop_id
        self.update()

    def selected_crop_id(self) -> str:
        return self._selected_id

    def request_repaint(self):
        """Owner calls this when zoom/pan changes so the overlay redraws
        without needing to re-push the crop list."""
        self.update()

    # ------------------------------------------------------------------
    # Coordinate transforms (source ↔ widget)
    # ------------------------------------------------------------------

    def _video_rect_in_widget(self) -> Optional[QRectF]:
        """Return the rectangle on this overlay covered by the rendered
        video (excluding letterbox bars). Returns None if the source dims
        aren't known yet."""
        geo = self._preview._get_render_geometry()
        if geo is None:
            return None
        widget_w, widget_h, _fw, _fh, _r, scaled_w, scaled_h = geo
        pan_x = self._preview._pan_x
        pan_y = self._preview._pan_y
        # Same formula as mpv's video positioning: center + pan, in
        # fractions of scaled video size.
        cx = widget_w / 2.0 + pan_x * scaled_w
        cy = widget_h / 2.0 + pan_y * scaled_h
        left = cx - scaled_w / 2.0
        top = cy - scaled_h / 2.0
        # The overlay is a top-level Qt.Tool window whose screen position
        # is pinned to the PreviewWidget's via sync_geometry_to_preview().
        # Windows can snap that HWND origin by ±1 px (DPI rounding / DWM
        # frame extension), which means overlay-local (0,0) is not
        # *exactly* preview-local (0,0). Strict QRectF.contains() then
        # fails for body / draw-new clicks even when handles still hit
        # via their ±6-px slack — matching the observed regression.
        # Round-trip the container's origin through global coords so the
        # returned rect lives in overlay-local space, independent of any
        # OS-level snapping between the two top-level coordinate systems.
        container_global = self._preview._container.mapToGlobal(QPoint(0, 0))
        container_in_overlay = self.mapFromGlobal(container_global)
        return QRectF(container_in_overlay.x() + left,
                      container_in_overlay.y() + top,
                      scaled_w, scaled_h)

    def _source_to_widget(self, x: int, y: int, w: int, h: int
                          ) -> Optional[QRectF]:
        vr = self._video_rect_in_widget()
        if vr is None or self._source_w <= 0 or self._source_h <= 0:
            return None
        sx = vr.width() / self._source_w
        sy = vr.height() / self._source_h
        return QRectF(vr.x() + x * sx,
                      vr.y() + y * sy,
                      max(1.0, w * sx),
                      max(1.0, h * sy))

    def _widget_to_source(self, pt: QPoint) -> Optional[Tuple[int, int]]:
        vr = self._video_rect_in_widget()
        if vr is None or self._source_w <= 0 or self._source_h <= 0:
            return None
        if vr.width() <= 0 or vr.height() <= 0:
            return None
        u = (pt.x() - vr.x()) / vr.width()
        v = (pt.y() - vr.y()) / vr.height()
        # Allow values slightly outside [0, 1] but clamp on commit. Map.
        sx = int(round(u * self._source_w))
        sy = int(round(v * self._source_h))
        return (sx, sy)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        # No `_edit_mode` gate here — outlines must paint when the
        # overlay is visible in view-only mode too. `_refresh_visibility`
        # owns the show/hide decision; if we got here the overlay is
        # showing and there's at least a clip context to draw against.
        if self._source_w <= 0 or self._source_h <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._edit_mode:
            # Layered top-level windows on Windows (Qt.Tool +
            # WA_TranslucentBackground) hit-test against painted pixel
            # alpha. Without a non-zero fill, only the crop outlines and
            # handle squares register as hit-able — clicks in the body
            # interior or empty video area fall through to the mpv HWND
            # below before mousePressEvent fires. alpha=1 is
            # imperceptible visually but makes the entire overlay capture
            # LMB, so draw / move / resize all reach the press handler.
            painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # Selection / handles concepts are edit-mode-only.
        selected = (self._find_crop(self._selected_id)
                    if self._edit_mode else None)
        sel_rect = (self._source_to_widget(
                        selected.x, selected.y, selected.w, selected.h)
                    if selected is not None else None)

        # Crop outlines. View-only mode hides inactive crops; edit mode
        # shows them as dashed so the user can find/re-enable them.
        for cr in self._crops:
            if not cr.active and not self._edit_mode:
                continue
            wrect = self._source_to_widget(cr.x, cr.y, cr.w, cr.h)
            if wrect is None:
                continue
            is_selected = self._edit_mode and (cr.id == self._selected_id)
            self._paint_crop_outline(
                painter, cr, wrect, selected=is_selected)

        # Resize handles for the selected crop (edit mode only).
        if self._edit_mode and selected is not None and sel_rect is not None:
            self._paint_handles(painter, sel_rect)

        # Live draw-new preview (edit mode only).
        if self._edit_mode and self._drag_kind == "draw":
            x0, y0 = self._draw_anchor_src
            x1, y1 = self._widget_to_source(self._last_drag_pos) or (x0, y0)
            sx, sy = min(x0, x1), min(y0, y1)
            sw, sh = abs(x1 - x0), abs(y1 - y0)
            if self._draw_aspect:
                sw, sh = self._apply_aspect_to_draw(sw, sh, self._draw_aspect)
            preview_rect = self._source_to_widget(sx, sy, max(1, sw), max(1, sh))
            if preview_rect is not None:
                pen = QPen(QColor("#5577aa"))
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(preview_rect)
        painter.end()

    def _paint_dim_outside(self, painter: QPainter, sel_rect: QRectF):
        vr = self._video_rect_in_widget()
        if vr is None:
            return
        # Dim every video pixel except the selected crop.
        dim = QColor(0, 0, 0, 130)
        painter.setBrush(QBrush(dim))
        painter.setPen(Qt.PenStyle.NoPen)
        # Top
        if sel_rect.top() > vr.top():
            painter.drawRect(QRectF(vr.left(), vr.top(),
                                    vr.width(), sel_rect.top() - vr.top()))
        # Bottom
        if sel_rect.bottom() < vr.bottom():
            painter.drawRect(QRectF(vr.left(), sel_rect.bottom(),
                                    vr.width(), vr.bottom() - sel_rect.bottom()))
        # Left
        if sel_rect.left() > vr.left():
            painter.drawRect(QRectF(vr.left(), sel_rect.top(),
                                    sel_rect.left() - vr.left(),
                                    sel_rect.height()))
        # Right
        if sel_rect.right() < vr.right():
            painter.drawRect(QRectF(sel_rect.right(), sel_rect.top(),
                                    vr.right() - sel_rect.right(),
                                    sel_rect.height()))

    def _paint_crop_outline(self, painter: QPainter, cr: CropRegion,
                            wrect: QRectF, selected: bool):
        color = QColor(self._group_color.get(cr.id, "#cccccc"))
        if not color.isValid():
            color = QColor("#cccccc")
        if selected:
            # White halo so the selected crop reads against bright frames
            # too — compensates for the (intentionally) absent dim layer.
            halo = QPen(QColor(255, 255, 255, 220))
            halo.setWidth(3)
            halo.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(halo)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wrect)
        pen = QPen(color)
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.SolidLine if cr.active
                     else Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(wrect)

    def _paint_handles(self, painter: QPainter, sel_rect: QRectF):
        painter.setPen(QPen(QColor("#5577aa"), 1))
        painter.setBrush(QBrush(QColor("#ffffff")))
        size = HANDLE_PX
        half = size / 2.0
        for hx, hy in self._handle_centers(sel_rect):
            painter.drawRect(QRectF(hx - half, hy - half, size, size))

    @staticmethod
    def _handle_centers(rect: QRectF):
        return (
            (rect.left(),  rect.top()),       # NW
            (rect.center().x(), rect.top()),  # N
            (rect.right(), rect.top()),       # NE
            (rect.right(), rect.center().y()),  # E
            (rect.right(), rect.bottom()),    # SE
            (rect.center().x(), rect.bottom()),  # S
            (rect.left(),  rect.bottom()),    # SW
            (rect.left(),  rect.center().y()),  # W
        )

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self._edit_mode or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        # Don't fight the preview's middle-mouse pan / wheel zoom — those
        # events go to the preview because our overlay is mouse-transparent
        # for non-edit-mode and only intercepts LMB anyway.
        pt = event.position().toPoint()
        self._last_drag_pos = pt
        # Hit-test handles of selected crop first.
        selected = self._find_crop(self._selected_id)
        if selected is not None:
            sel_rect = self._source_to_widget(
                selected.x, selected.y, selected.w, selected.h)
            if sel_rect is not None:
                handle = self._handle_hit(pt, sel_rect)
                if handle:
                    self._begin_resize(selected, handle, pt)
                    event.accept()
                    return
                if sel_rect.contains(pt):
                    self._begin_move(selected, pt)
                    event.accept()
                    return
        # Hit-test any crop body (topmost wins; iterate reversed).
        for cr in reversed(self._crops):
            wrect = self._source_to_widget(cr.x, cr.y, cr.w, cr.h)
            if wrect is not None and wrect.contains(pt):
                self.crop_selected.emit(cr.id)
                self._selected_id = cr.id
                self._begin_move(cr, pt)
                self.update()
                event.accept()
                return
        # Empty area inside video → start draw-new.
        vr = self._video_rect_in_widget()
        if vr is not None and vr.contains(pt):
            self._begin_draw(pt)
            # Deselect any current selection.
            if self._selected_id:
                self._selected_id = ""
                self.crop_selected.emit("")
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._edit_mode:
            super().mouseMoveEvent(event)
            return
        pt = event.position().toPoint()
        self._last_drag_pos = pt
        if self._drag_kind == "draw":
            self.update()
            return
        if self._drag_kind == "move":
            self._do_move(pt)
            return
        if self._drag_kind == "resize":
            self._do_resize(pt)
            return
        # Hover cursor feedback.
        cursor = self._cursor_for_position(pt)
        self.setCursor(cursor)

    def mouseReleaseEvent(self, event):
        if not self._edit_mode or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._drag_kind == "draw":
            self._commit_draw()
        elif self._drag_kind == "move" or self._drag_kind == "resize":
            self._commit_geometry_change()
        self._drag_kind = ""
        self._drag_handle = ""
        self._drag_crop_id = ""
        self.unsetCursor()
        event.accept()

    def keyPressEvent(self, event):
        if not self._edit_mode:
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_id:
                self.delete_requested.emit(self._selected_id)
                event.accept()
                return
        if event.key() == Qt.Key.Key_Escape:
            if self._selected_id:
                self._selected_id = ""
                self.crop_selected.emit("")
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Hit-test helpers
    # ------------------------------------------------------------------

    def _find_crop(self, crop_id: str) -> Optional[CropRegion]:
        if not crop_id:
            return None
        for cr in self._crops:
            if cr.id == crop_id:
                return cr
        return None

    def _handle_hit(self, pt: QPoint, sel_rect: QRectF) -> str:
        size = HANDLE_PX
        half = size / 2.0
        for h, (hx, hy) in zip(_HANDLES, self._handle_centers(sel_rect)):
            if (abs(pt.x() - hx) <= half + 1
                    and abs(pt.y() - hy) <= half + 1):
                return h
        return ""

    def _cursor_for_position(self, pt: QPoint) -> QCursor:
        selected = self._find_crop(self._selected_id)
        if selected is not None:
            sel_rect = self._source_to_widget(
                selected.x, selected.y, selected.w, selected.h)
            if sel_rect is not None:
                handle = self._handle_hit(pt, sel_rect)
                if handle:
                    return QCursor(_HANDLE_CURSORS[handle])
                if sel_rect.contains(pt):
                    return QCursor(Qt.CursorShape.SizeAllCursor)
        for cr in self._crops:
            wrect = self._source_to_widget(cr.x, cr.y, cr.w, cr.h)
            if wrect is not None and wrect.contains(pt):
                return QCursor(Qt.CursorShape.PointingHandCursor)
        vr = self._video_rect_in_widget()
        if vr is not None and vr.contains(pt):
            return QCursor(Qt.CursorShape.CrossCursor)
        return QCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Drag begin / commit
    # ------------------------------------------------------------------

    def _begin_draw(self, pt: QPoint):
        src = self._widget_to_source(pt)
        if src is None:
            return
        # Clamp anchor to source bounds.
        sx = max(0, min(self._source_w - 1, src[0]))
        sy = max(0, min(self._source_h - 1, src[1]))
        self._drag_kind = "draw"
        self._draw_anchor_src = (sx, sy)
        # If a previously-selected crop had an aspect lock, inherit it for
        # the new draw too (user can reset via the panel).
        sel = self._find_crop(self._selected_id)
        self._draw_aspect = resolve_aspect(sel) if sel is not None else None

    def _begin_move(self, crop: CropRegion, pt: QPoint):
        self._drag_kind = "move"
        self._drag_crop_id = crop.id
        self._drag_start_widget = pt
        self._drag_start_rect = (crop.x, crop.y, crop.w, crop.h)

    def _begin_resize(self, crop: CropRegion, handle: str, pt: QPoint):
        self._drag_kind = "resize"
        self._drag_crop_id = crop.id
        self._drag_handle = handle
        self._drag_start_widget = pt
        self._drag_start_rect = (crop.x, crop.y, crop.w, crop.h)

    def _do_move(self, pt: QPoint):
        crop = self._find_crop(self._drag_crop_id)
        if crop is None:
            return
        # Convert delta in widget space → source space via the video rect.
        delta_src = self._delta_widget_to_source(
            pt - self._drag_start_widget)
        if delta_src is None:
            return
        dx, dy = delta_src
        x0, y0, w, h = self._drag_start_rect
        nx = max(0, min(self._source_w - w, x0 + dx))
        ny = max(0, min(self._source_h - h, y0 + dy))
        if nx == crop.x and ny == crop.y:
            return
        crop.x = int(nx)
        crop.y = int(ny)
        self.update()

    def _do_resize(self, pt: QPoint):
        crop = self._find_crop(self._drag_crop_id)
        if crop is None:
            return
        delta_src = self._delta_widget_to_source(
            pt - self._drag_start_widget)
        if delta_src is None:
            return
        dx, dy = delta_src
        x0, y0, w0, h0 = self._drag_start_rect
        x1, y1, x2, y2 = x0, y0, x0 + w0, y0 + h0
        h = self._drag_handle
        if h in (HIT_NW, HIT_W, HIT_SW):
            x1 = min(x2 - MIN_CROP_SIZE, max(0, x0 + dx))
        if h in (HIT_NW, HIT_N, HIT_NE):
            y1 = min(y2 - MIN_CROP_SIZE, max(0, y0 + dy))
        if h in (HIT_NE, HIT_E, HIT_SE):
            x2 = max(x1 + MIN_CROP_SIZE,
                     min(self._source_w, x0 + w0 + dx))
        if h in (HIT_SW, HIT_S, HIT_SE):
            y2 = max(y1 + MIN_CROP_SIZE,
                     min(self._source_h, y0 + h0 + dy))
        nw, nh = x2 - x1, y2 - y1
        # Aspect-ratio lock: snap one axis to honour the ratio. Pivot
        # depends on the handle so the un-dragged corner stays put.
        aspect = resolve_aspect(crop)
        if aspect is not None and nw > 0 and nh > 0:
            ar = aspect[0] / aspect[1]
            # Prefer adjusting the axis the handle drags; for corners,
            # take whichever yields the *larger* enclosing rect so we
            # don't shrink the active axis.
            if h in (HIT_N, HIT_S):
                nw = int(round(nh * ar))
            elif h in (HIT_E, HIT_W):
                nh = int(round(nw / ar))
            else:
                # Corner: keep the dragged extent, snap the other axis.
                want_w = int(round(nh * ar))
                want_h = int(round(nw / ar))
                if want_w >= nw:
                    nw = want_w
                else:
                    nh = want_h
            nw = max(MIN_CROP_SIZE, min(self._source_w, nw))
            nh = max(MIN_CROP_SIZE, min(self._source_h, nh))
            # Re-anchor based on which corner is pinned.
            if h in (HIT_NW, HIT_W, HIT_SW):
                x1 = x2 - nw
            else:
                x2 = x1 + nw
            if h in (HIT_NW, HIT_N, HIT_NE):
                y1 = y2 - nh
            else:
                y2 = y1 + nh
            # Clamp to source bounds.
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > self._source_w:
                x1 -= (x2 - self._source_w)
                x2 = self._source_w
            if y2 > self._source_h:
                y1 -= (y2 - self._source_h)
                y2 = self._source_h
            nw, nh = x2 - x1, y2 - y1
        if nw < MIN_CROP_SIZE or nh < MIN_CROP_SIZE:
            return
        crop.x = int(x1)
        crop.y = int(y1)
        crop.w = int(nw)
        crop.h = int(nh)
        self.update()

    def _delta_widget_to_source(self, delta: QPoint) -> Optional[Tuple[int, int]]:
        vr = self._video_rect_in_widget()
        if vr is None or vr.width() <= 0 or vr.height() <= 0:
            return None
        sx = delta.x() * (self._source_w / vr.width())
        sy = delta.y() * (self._source_h / vr.height())
        return (int(round(sx)), int(round(sy)))

    def _apply_aspect_to_draw(self, w: int, h: int,
                              aspect: Tuple[int, int]) -> Tuple[int, int]:
        # Snap to the larger axis so the user's drag direction wins.
        ar = aspect[0] / aspect[1]
        if w / max(1, h) > ar:
            return (w, max(1, int(round(w / ar))))
        return (max(1, int(round(h * ar))), h)

    def _commit_draw(self):
        if not hasattr(self, "_last_drag_pos"):
            return
        x0, y0 = self._draw_anchor_src
        end_src = self._widget_to_source(self._last_drag_pos)
        if end_src is None:
            return
        x1, y1 = end_src
        sx, sy = min(x0, x1), min(y0, y1)
        sw, sh = abs(x1 - x0), abs(y1 - y0)
        if self._draw_aspect:
            sw, sh = self._apply_aspect_to_draw(sw, sh, self._draw_aspect)
        # Clamp to source bounds.
        sx = max(0, min(self._source_w - MIN_CROP_SIZE, sx))
        sy = max(0, min(self._source_h - MIN_CROP_SIZE, sy))
        sw = max(MIN_CROP_SIZE, min(self._source_w - sx, sw))
        sh = max(MIN_CROP_SIZE, min(self._source_h - sy, sh))
        if sw < MIN_CROP_SIZE or sh < MIN_CROP_SIZE:
            return
        self.new_crop_drawn.emit(int(sx), int(sy), int(sw), int(sh))

    def _commit_geometry_change(self):
        crop = self._find_crop(self._drag_crop_id)
        if crop is None:
            return
        # Capture the new values, then revert the live crop to its pre-drag
        # rect. That way the model's undo snapshot inside
        # ``update_crop_region`` captures the BEFORE state — the receiver
        # then applies the new values atomically as a single undo entry.
        nx, ny, nw, nh = crop.x, crop.y, crop.w, crop.h
        sx, sy, sw, sh = self._drag_start_rect
        if (nx, ny, nw, nh) == (sx, sy, sw, sh):
            return
        crop.x, crop.y, crop.w, crop.h = sx, sy, sw, sh
        self.crop_geometry_changed.emit(
            crop.id, int(nx), int(ny), int(nw), int(nh))

    def _cancel_drag(self):
        self._drag_kind = ""
        self._drag_crop_id = ""
        self._drag_handle = ""
