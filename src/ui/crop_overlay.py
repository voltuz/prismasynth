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
        super().__init__(parent or preview)
        self._preview = preview
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                          True)
        self.setVisible(False)
        # We want keyboard input for Delete / Escape only when edit mode is
        # on; we set this dynamically.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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

    def set_edit_mode(self, on: bool, clip_id: Optional[str] = None,
                      source_w: int = 0, source_h: int = 0,
                      crops: Optional[List[CropRegion]] = None,
                      group_colors: Optional[dict] = None):
        """Toggle edit mode. When ``on`` is True, ``clip_id`` / source
        dims / crops must be valid. When False, the overlay clears its
        state so it doesn't leak between clips."""
        self._edit_mode = bool(on)
        if on:
            self._clip_id = clip_id
            self._source_w = int(source_w)
            self._source_h = int(source_h)
            self._crops = list(crops or [])
            self._group_color = dict(group_colors or {})
        else:
            self._clip_id = None
            self._source_w = 0
            self._source_h = 0
            self._crops = []
            self._selected_id = ""
            self._cancel_drag()
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on)
        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus if on else Qt.FocusPolicy.NoFocus)
        self.setVisible(on)
        if on:
            self.raise_()
            self.setFocus()
        self.update()

    def set_crops(self, crops: List[CropRegion],
                  group_colors: Optional[dict] = None):
        """Refresh the crop list (e.g. on clips_changed). Preserves the
        current selection if the id still exists."""
        self._crops = list(crops)
        if group_colors is not None:
            self._group_color = dict(group_colors)
        if self._selected_id and not any(
                cr.id == self._selected_id for cr in self._crops):
            self._selected_id = ""
        self.update()

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
        # The overlay widget's geometry covers the whole PreviewWidget;
        # the video lives inside `_container` which already has 0,0 origin
        # in this widget's coordinates because both are children of
        # PreviewWidget with matching position. mapFromGlobal would be
        # overkill — we use the container's geometry as offset.
        cont_geo = self._preview._container.geometry()
        return QRectF(cont_geo.x() + left,
                      cont_geo.y() + top,
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
        if not self._edit_mode or self._source_w <= 0 or self._source_h <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Find the selected crop and its widget rect for the dim overlay.
        selected = self._find_crop(self._selected_id)
        if selected is not None:
            sel_rect = self._source_to_widget(
                selected.x, selected.y, selected.w, selected.h)
        else:
            sel_rect = None

        # Dim overlay outside the selected crop (only when something is
        # selected — keeps the unselected case readable).
        if sel_rect is not None:
            self._paint_dim_outside(painter, sel_rect)

        # Each crop's outline.
        for cr in self._crops:
            wrect = self._source_to_widget(cr.x, cr.y, cr.w, cr.h)
            if wrect is None:
                continue
            self._paint_crop_outline(
                painter, cr, wrect, selected=(cr.id == self._selected_id))

        # Resize handles for the selected crop.
        if selected is not None and sel_rect is not None:
            self._paint_handles(painter, sel_rect)

        # Live draw-new preview.
        if self._drag_kind == "draw":
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
        pen = QPen(color)
        pen.setWidth(2 if selected else 1)
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
