"""Detachable graph editor for animating CropRegion keyframes.

Hosted as a ``QDockWidget`` on MainWindow but defaults to floating so it
opens as a standalone window (and can be parked on a second monitor).
The inner ``KeyframeGraph`` widget is a custom-painted Cartesian graph
that visualises and edits the four axis tracks (``x``, ``y``, ``w``,
``h``) of one CropRegion at a time.

Interaction model:
- Click a keyframe dot → select it (Ctrl to add to selection).
- Drag a selected dot → move it (snaps X to integer source frame).
- Click empty curve area → add a keyframe at the clicked frame to
  the curve nearest the click (using its interpolated value).
- Right-click a keyframe → context menu (Delete · Linear · Bezier · Step).
- Bezier handle dots appear next to bezier-interp keys and can be
  dragged to shape the tangent.
- Wheel zooms the X axis; middle-mouse pans.
- A vertical playhead line follows the main timeline's playhead in
  source-frame space.

All mutations route through ``TimelineModel`` so the global undo stack
covers every keyframe edit. Drag operations only commit on release so a
single drag is one undo unit instead of dozens of intermediate states.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QPolygon,
)
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QHBoxLayout, QLabel, QMenu, QPushButton,
    QVBoxLayout, QWidget,
)

from core.crop_region import CropRegion
from core.keyframe import (
    INTERP_BEZIER, INTERP_LINEAR, INTERP_STEP, Keyframe, KeyframeTrack,
)
from core.timeline import TimelineModel


# Track display config: axis-name → (color, display label).
_TRACK_COLORS: Dict[str, Tuple[str, str]] = {
    "x": ("#e8525c", "X"),
    "y": ("#5fc66a", "Y"),
    "w": ("#5dc9e0", "W"),
    "h": ("#e8c247", "H"),
}
_AXES = ("x", "y", "w", "h")

# Visual constants
_DOT_RADIUS = 5
_HANDLE_RADIUS = 3
_HANDLE_REACH_PX = 40.0   # default handle length in pixels at zoom 1
_MARGIN_X = 40
_MARGIN_Y = 16
_BG_COLOR = QColor("#1c1c1c")
_GRID_COLOR = QColor("#2c2c2c")
_AXIS_COLOR = QColor("#444444")
_PLAYHEAD_COLOR = QColor("#dd6633")
_HIT_TOL_PX = 8.0  # dot pick tolerance


def _luminance_text_color(bg_hex: str) -> QColor:
    """Pick a readable text color for a colored chip background."""
    c = QColor(bg_hex)
    y = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
    return QColor("#ffffff") if y < 0.55 else QColor("#000000")


class KeyframeGraph(QWidget):
    """Custom-painted graph that edits one CropRegion's tracks."""

    # Emitted when the user drags the playhead marker.
    playhead_scrub_requested = Signal(int)  # source_frame
    # Emitted on every live drag mutation so the owner can repaint the
    # crop overlay in real time (without a heavy clips_changed rebuild).
    live_edit_changed = Signal()

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._clip_id: Optional[str] = None
        self._crop_id: Optional[str] = None
        # Source fps of the targeted crop's clip — used to size the
        # 81-frame@16fps export-segment bands. 0 = unknown (no bands).
        self._src_fps: float = 0.0
        # Source-frame range visible in the X axis.
        self._x_min: float = 0.0
        self._x_max: float = 100.0
        # Per-track visibility.
        self._visible: Dict[str, bool] = {a: True for a in _AXES}
        # Selection: list of (axis, source_frame).
        self._selection: List[Tuple[str, int]] = []
        # Playhead in source-frame space.
        self._playhead: int = -1
        # Drag state.
        self._drag_kind: str = ""  # "" | "key" | "handle" | "pan" | "scrub"
        self._drag_origin_widget: QPoint = QPoint()
        # Live drag: we mutate the model keyframes directly so the curve
        # and the preview crop update in real time, holding the actual
        # Keyframe object refs (valid because no clips_changed/snapshot
        # fires mid-drag). One undo snapshot per drag, pushed lazily on
        # the first effective change.
        # Each entry: (axis, track, key_obj, start_frame, start_value).
        self._drag_keys: List[tuple] = []
        # Per-axis value-units-per-pixel captured at drag start so the
        # vertical mapping stays linear even though the keys (and thus
        # the track value range) mutate live during the drag.
        self._drag_value_per_px: Dict[str, float] = {}
        # (axis, track, key_obj, side, start_in, start_out)
        self._drag_handle: Optional[tuple] = None
        self._drag_pushed_undo: bool = False
        self.setMouseTracking(True)
        self.setMinimumHeight(180)
        self.setMinimumWidth(360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # React to timeline mutations so the graph reflects undo / autokey.
        self._timeline.clips_changed.connect(self.update)

    # --- Public API -----------------------------------------------------

    def set_target(self, clip_id: Optional[str], crop_id: Optional[str],
                   src_fps: float = 0.0):
        """Point the editor at a specific (clip, crop). ``None`` clears.
        ``src_fps`` sizes the export-segment bands."""
        self._clip_id = clip_id
        self._crop_id = crop_id
        self._src_fps = float(src_fps)
        self._selection.clear()
        self._cancel_drag()
        self._fit_x_axis()
        self.update()

    def set_track_visible(self, axis: str, visible: bool):
        if axis not in _AXES:
            return
        self._visible[axis] = bool(visible)
        self.update()

    def set_playhead(self, source_frame: int):
        if source_frame == self._playhead:
            return
        self._playhead = int(source_frame)
        self.update()

    def fit_x_axis(self):
        self._fit_x_axis()
        self.update()

    # --- Crop / track lookup --------------------------------------------

    def _current_crop(self) -> Optional[CropRegion]:
        if not self._clip_id or not self._crop_id:
            return None
        clip = self._timeline.get_clip_by_id(self._clip_id)
        if clip is None or clip.is_gap:
            return None
        for cr in clip.crop_regions:
            if cr.id == self._crop_id:
                return cr
        return None

    def _track_for(self, cr: CropRegion, axis: str) -> KeyframeTrack:
        return cr.track_for(axis)

    def _track_value_range(self, cr: CropRegion, axis: str) -> Tuple[float, float]:
        """Return (lo, hi) for the per-axis Y mapping. Includes the
        static base value so an empty track still has a sensible span.
        At least 4-unit minimum span so a single-key track doesn't
        collapse to a horizontal line clipped to one pixel."""
        track = self._track_for(cr, axis)
        vals: List[float] = [float(cr.base_value(axis))]
        for k in track.keys:
            vals.append(float(k.value))
        lo, hi = min(vals), max(vals)
        if hi - lo < 4:
            mid = (lo + hi) / 2.0
            lo, hi = mid - 2, mid + 2
        pad = (hi - lo) * 0.1
        return (lo - pad, hi + pad)

    def _fit_x_axis(self):
        """Fit the X range to the clip's full source-frame span (or
        fall back to the keys' span when the clip is unavailable)."""
        if self._clip_id:
            clip = self._timeline.get_clip_by_id(self._clip_id)
            if clip is not None and not clip.is_gap:
                self._x_min = float(clip.source_in)
                self._x_max = float(clip.source_out + 1)
                return
        cr = self._current_crop()
        if cr is None:
            self._x_min, self._x_max = 0.0, 100.0
            return
        frames = []
        for ax in _AXES:
            for k in cr.track_for(ax).keys:
                frames.append(k.source_frame)
        if not frames:
            self._x_min, self._x_max = 0.0, 100.0
            return
        lo, hi = min(frames), max(frames)
        if hi - lo < 10:
            mid = (lo + hi) / 2
            lo, hi = mid - 5, mid + 5
        pad = (hi - lo) * 0.1
        self._x_min, self._x_max = float(lo - pad), float(hi + pad)

    # --- Coordinate mapping ---------------------------------------------

    def _plot_rect(self) -> QRect:
        return QRect(_MARGIN_X, _MARGIN_Y,
                     max(1, self.width() - _MARGIN_X - 6),
                     max(1, self.height() - _MARGIN_Y * 2))

    def _frame_to_px(self, frame: float) -> float:
        r = self._plot_rect()
        span = max(1e-6, self._x_max - self._x_min)
        return r.left() + (frame - self._x_min) / span * r.width()

    def _px_to_frame(self, px: float) -> float:
        r = self._plot_rect()
        span = self._x_max - self._x_min
        if r.width() <= 0:
            return self._x_min
        return self._x_min + (px - r.left()) / r.width() * span

    def _value_to_px(self, cr: CropRegion, axis: str, value: float) -> float:
        r = self._plot_rect()
        lo, hi = self._track_value_range(cr, axis)
        span = max(1e-6, hi - lo)
        # Y axis flipped so higher values render higher up.
        return r.bottom() - (value - lo) / span * r.height()

    def _px_to_value(self, cr: CropRegion, axis: str, py: float) -> float:
        r = self._plot_rect()
        lo, hi = self._track_value_range(cr, axis)
        if r.height() <= 0:
            return lo
        return lo + (r.bottom() - py) / r.height() * (hi - lo)

    # --- Paint ----------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG_COLOR)
        self._paint_grid(painter)

        cr = self._current_crop()
        if cr is None:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No crop selected")
            painter.end()
            return

        self._paint_segment_bands(painter, cr)
        self._paint_playhead(painter)
        for axis in _AXES:
            if not self._visible[axis]:
                continue
            self._paint_track(painter, cr, axis)
        self._paint_axis_labels(painter, cr)
        painter.end()

    def _paint_segment_bands(self, painter: QPainter, cr: CropRegion):
        """Shade each export segment's source-frame window so the user
        sees which slices of the animation actually get exported.
        Read-only. Needs ``_src_fps`` to size the 81-frame@16fps window."""
        if self._src_fps <= 0:
            return
        from core.crop_region import required_source_frames
        req = required_source_frames(self._src_fps)
        r = self._plot_rect()
        for seg in cr.segments:
            x0 = self._frame_to_px(seg.anchor_frame)
            x1 = self._frame_to_px(seg.anchor_frame + req)
            # Clip to the plot rect horizontally.
            lo = max(r.left(), min(x0, x1))
            hi = min(r.right(), max(x0, x1))
            if hi <= lo:
                continue
            alpha = 46 if seg.active else 20
            painter.fillRect(int(lo), r.top(), int(hi - lo), r.height(),
                             QColor(232, 167, 53, alpha))

    def _paint_grid(self, painter: QPainter):
        r = self._plot_rect()
        painter.fillRect(r, QColor("#181818"))
        painter.setPen(QPen(_GRID_COLOR, 1, Qt.PenStyle.DotLine))
        # Vertical gridlines at every ~80px.
        n_ticks = max(2, r.width() // 80)
        for i in range(n_ticks + 1):
            x = r.left() + i * r.width() / n_ticks
            painter.drawLine(int(x), r.top(), int(x), r.bottom())
            frame = self._px_to_frame(x)
            painter.setPen(QColor("#888"))
            painter.drawText(int(x) - 20, r.top() - 2, f"{int(frame)}")
            painter.setPen(QPen(_GRID_COLOR, 1, Qt.PenStyle.DotLine))
        painter.setPen(QPen(_AXIS_COLOR, 1))
        painter.drawRect(r)

    def _paint_playhead(self, painter: QPainter):
        if self._playhead < 0:
            return
        x = self._frame_to_px(self._playhead)
        r = self._plot_rect()
        if not (r.left() - 1 <= x <= r.right() + 1):
            return
        painter.setPen(QPen(_PLAYHEAD_COLOR, 1))
        painter.drawLine(int(x), r.top(), int(x), r.bottom())

    def _paint_track(self, painter: QPainter, cr: CropRegion, axis: str):
        color = QColor(_TRACK_COLORS[axis][0])
        track = self._track_for(cr, axis)
        # The curve is the *displayed* sample, so callers see the same
        # value the overlay uses at any frame. We sample the track at
        # every visible pixel column.
        r = self._plot_rect()
        if track:
            painter.setPen(QPen(color, 2))
            # Walk per-segment so we honour the actual interpolation
            # type instead of approximating bezier with line segments
            # of unknown density.
            if len(track.keys) == 1:
                # Flat line — but only within [x_min, x_max].
                v = track.keys[0].value
                py = self._value_to_px(cr, axis, v)
                painter.drawLine(r.left(), int(py), r.right(), int(py))
            else:
                for i in range(len(track.keys) - 1):
                    a = track.keys[i]
                    b = track.keys[i + 1]
                    self._paint_segment(painter, cr, axis, a, b)
                # Pre / post extrapolation = clamp to nearest key.
                first = track.keys[0]
                last = track.keys[-1]
                py0 = self._value_to_px(cr, axis, first.value)
                pyN = self._value_to_px(cr, axis, last.value)
                px0 = self._frame_to_px(first.source_frame)
                pxN = self._frame_to_px(last.source_frame)
                painter.drawLine(r.left(), int(py0), int(px0), int(py0))
                painter.drawLine(int(pxN), int(pyN), r.right(), int(pyN))
        else:
            # No keys — show the static base as a dashed line so the
            # user sees what value the track will use if they add a key.
            base = float(cr.base_value(axis))
            py = self._value_to_px(cr, axis, base)
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(r.left(), int(py), r.right(), int(py))

        # Keyframe dots + handles.
        for k in track.keys:
            self._paint_key_dot(painter, cr, axis, k, color)

    def _paint_segment(self, painter: QPainter, cr: CropRegion, axis: str,
                       a: Keyframe, b: Keyframe):
        if a.interp == INTERP_STEP:
            x0 = self._frame_to_px(a.source_frame)
            y0 = self._value_to_px(cr, axis, a.value)
            x1 = self._frame_to_px(b.source_frame)
            painter.drawLine(int(x0), int(y0), int(x1), int(y0))
            y1 = self._value_to_px(cr, axis, b.value)
            painter.drawLine(int(x1), int(y0), int(x1), int(y1))
            return
        # Linear or bezier: sample at every visible pixel between
        # x(a) and x(b) and stitch line segments.
        x_lo = self._frame_to_px(a.source_frame)
        x_hi = self._frame_to_px(b.source_frame)
        if x_hi - x_lo < 2:
            painter.drawLine(int(x_lo),
                             int(self._value_to_px(cr, axis, a.value)),
                             int(x_hi),
                             int(self._value_to_px(cr, axis, b.value)))
            return
        track = self._track_for(cr, axis)
        steps = max(2, int(x_hi - x_lo))
        last_x = int(x_lo)
        last_y = int(self._value_to_px(cr, axis, a.value))
        for s in range(1, steps + 1):
            px = x_lo + (x_hi - x_lo) * s / steps
            frame = self._px_to_frame(px)
            v = track.sample(frame)
            if v is None:
                continue
            py = self._value_to_px(cr, axis, v)
            painter.drawLine(last_x, last_y, int(px), int(py))
            last_x, last_y = int(px), int(py)

    def _paint_key_dot(self, painter: QPainter, cr: CropRegion,
                       axis: str, k: Keyframe, color: QColor):
        # Drags mutate the model key directly, so reading source_frame /
        # value here already reflects an in-progress edit.
        x = self._frame_to_px(k.source_frame)
        y = self._value_to_px(cr, axis, k.value)
        selected = (axis, k.source_frame) in self._selection
        if selected:
            painter.setPen(QPen(QColor("#ffffff"), 2))
        else:
            painter.setPen(QPen(QColor("#202020"), 1))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPoint(int(x), int(y)),
                            _DOT_RADIUS, _DOT_RADIUS)

        # Bezier handles only render for selected bezier keys.
        if selected and k.interp == INTERP_BEZIER:
            self._paint_handles(painter, cr, axis, k, color, x, y)

    def _paint_handles(self, painter: QPainter, cr: CropRegion,
                       axis: str, k: Keyframe, color: QColor,
                       x: float, y: float):
        # Convert the handle's (frame, value) offsets to pixel-space.
        # Drags mutate k.in_handle / k.out_handle live, so reading them
        # straight from the model already reflects an in-progress edit.
        dx_in = k.in_handle[0]
        dy_in = k.in_handle[1]
        dx_out = k.out_handle[0]
        dy_out = k.out_handle[1]
        # Each handle is offset in (frame, value) space — convert via
        # the per-track Y range so the handle pixel offset isn't tied
        # to value magnitude.
        in_x = self._frame_to_px(k.source_frame + dx_in)
        in_y = self._value_to_px(cr, axis, k.value + dy_in)
        out_x = self._frame_to_px(k.source_frame + dx_out)
        out_y = self._value_to_px(cr, axis, k.value + dy_out)
        painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(x), int(y), int(in_x), int(in_y))
        painter.drawLine(int(x), int(y), int(out_x), int(out_y))
        painter.setPen(QPen(QColor("#202020"), 1))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(QPoint(int(in_x), int(in_y)),
                            _HANDLE_RADIUS, _HANDLE_RADIUS)
        painter.drawEllipse(QPoint(int(out_x), int(out_y)),
                            _HANDLE_RADIUS, _HANDLE_RADIUS)

    def _paint_axis_labels(self, painter: QPainter, cr: CropRegion):
        # Color chips for visible tracks along the top-left.
        x = 6
        y = 4
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        for axis in _AXES:
            if not self._visible[axis]:
                continue
            color = QColor(_TRACK_COLORS[axis][0])
            painter.fillRect(x, y, 12, 12, color)
            painter.setPen(_luminance_text_color(_TRACK_COLORS[axis][0]))
            painter.drawText(x + 2, y + 10, _TRACK_COLORS[axis][1])
            x += 22

    # --- Hit testing ----------------------------------------------------

    def _hit_test_dot(self, pt: QPoint) -> Optional[Tuple[str, int]]:
        cr = self._current_crop()
        if cr is None:
            return None
        best = None
        best_d2 = _HIT_TOL_PX * _HIT_TOL_PX
        for axis in _AXES:
            if not self._visible[axis]:
                continue
            track = self._track_for(cr, axis)
            for k in track.keys:
                kx = self._frame_to_px(k.source_frame)
                ky = self._value_to_px(cr, axis, k.value)
                dx = pt.x() - kx
                dy = pt.y() - ky
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best = (axis, k.source_frame)
                    best_d2 = d2
        return best

    def _hit_test_handle(self, pt: QPoint
                         ) -> Optional[Tuple[str, int, str]]:
        cr = self._current_crop()
        if cr is None:
            return None
        # Only handles on currently-selected bezier keys are clickable.
        for axis, sf in self._selection:
            track = self._track_for(cr, axis)
            k = track.find_key(sf)
            if k is None or k.interp != INTERP_BEZIER:
                continue
            for side, (dx_off, dy_off) in (
                    ("in", k.in_handle), ("out", k.out_handle)):
                hx = self._frame_to_px(sf + dx_off)
                hy = self._value_to_px(cr, axis, k.value + dy_off)
                if (abs(pt.x() - hx) <= _HANDLE_RADIUS + 3
                        and abs(pt.y() - hy) <= _HANDLE_RADIUS + 3):
                    return (axis, sf, side)
        return None

    def _hit_test_curve(self, pt: QPoint) -> Optional[str]:
        """Return the axis whose curve passes within a few pixels of
        ``pt`` (in screen space). Used by empty-area click to decide
        which track to add a key to."""
        cr = self._current_crop()
        if cr is None:
            return None
        best_axis = None
        best_d = 10  # tight pixel tolerance
        frame = self._px_to_frame(pt.x())
        for axis in _AXES:
            if not self._visible[axis]:
                continue
            track = self._track_for(cr, axis)
            v = track.sample(frame)
            if v is None:
                v = float(cr.base_value(axis))
            py = self._value_to_px(cr, axis, v)
            d = abs(py - pt.y())
            if d < best_d:
                best_d = d
                best_axis = axis
        return best_axis

    def _playhead_hit(self, pt: QPoint) -> bool:
        if self._playhead < 0:
            return False
        x = self._frame_to_px(self._playhead)
        return abs(pt.x() - x) <= 4 and self._plot_rect().contains(pt)

    # --- Mouse ----------------------------------------------------------

    def mousePressEvent(self, event):
        pt = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_kind = "pan"
            self._drag_origin_widget = pt
            self._pan_x_min_at_start = self._x_min
            self._pan_x_max_at_start = self._x_max
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # Handle drag (bezier).
            hh = self._hit_test_handle(pt)
            if hh is not None:
                axis, sf, side = hh
                cr = self._current_crop()
                if cr is not None:
                    track = self._track_for(cr, axis)
                    k = track.find_key(sf)
                    if k is not None:
                        self._drag_handle = (axis, track, k, side,
                                             k.in_handle, k.out_handle)
                        self._drag_pushed_undo = False
                        self._drag_kind = "handle"
                        self._drag_origin_widget = pt
                        event.accept()
                        return
            # Dot hit.
            hit = self._hit_test_dot(pt)
            if hit is not None:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if hit in self._selection:
                        self._selection.remove(hit)
                    else:
                        self._selection.append(hit)
                else:
                    if hit not in self._selection:
                        self._selection = [hit]
                self._begin_key_drag(pt)
                self.update()
                event.accept()
                return
            # Playhead grab.
            if self._playhead_hit(pt):
                self._drag_kind = "scrub"
                event.accept()
                return
            # Empty area: add a key to the nearest visible curve.
            axis = self._hit_test_curve(pt)
            if axis is not None:
                self._add_key_at(axis, pt)
                event.accept()
                return
            # Clicked in dead space → clear selection.
            if self._selection:
                self._selection.clear()
                self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pt = event.position().toPoint()
        if self._drag_kind == "pan":
            dx_px = pt.x() - self._drag_origin_widget.x()
            r = self._plot_rect()
            if r.width() > 0:
                span = self._pan_x_max_at_start - self._pan_x_min_at_start
                dx_frame = -dx_px / r.width() * span
                self._x_min = self._pan_x_min_at_start + dx_frame
                self._x_max = self._pan_x_max_at_start + dx_frame
            self.update()
            event.accept()
            return
        if self._drag_kind == "scrub":
            frame = max(0, int(round(self._px_to_frame(pt.x()))))
            self.playhead_scrub_requested.emit(frame)
            event.accept()
            return
        if self._drag_kind == "key":
            self._do_key_drag(pt)
            event.accept()
            return
        if self._drag_kind == "handle":
            self._do_handle_drag(pt)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_kind == "key":
            self._commit_key_drag()
        elif self._drag_kind == "handle":
            self._commit_handle_drag()
        self._drag_kind = ""
        self._drag_origin_widget = QPoint()
        self._drag_handle = None
        self._drag_keys = []
        self._drag_pushed_undo = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        pt = event.pos()
        hit = self._hit_test_dot(pt)
        if hit is None:
            super().contextMenuEvent(event)
            return
        if hit not in self._selection:
            self._selection = [hit]
            self.update()
        menu = QMenu(self)
        del_act = QAction("Delete", self)
        del_act.triggered.connect(self._delete_selected_keys)
        menu.addAction(del_act)
        menu.addSeparator()
        for interp_id, label in (
                (INTERP_LINEAR, "Linear"),
                (INTERP_BEZIER, "Bezier"),
                (INTERP_STEP, "Step (hold)")):
            act = QAction(label, self)
            act.triggered.connect(
                lambda _checked=False, iid=interp_id:
                self._set_selected_interp(iid))
            menu.addAction(act)
        menu.exec(event.globalPos())

    def wheelEvent(self, event):
        # Anchor zoom around the cursor's frame.
        cursor_frame = self._px_to_frame(event.position().x())
        factor = 1.25 if event.angleDelta().y() < 0 else 1 / 1.25
        new_min = cursor_frame - (cursor_frame - self._x_min) * factor
        new_max = cursor_frame + (self._x_max - cursor_frame) * factor
        if new_max - new_min < 4:
            return
        self._x_min, self._x_max = new_min, new_max
        self.update()

    # --- Drag implementations ------------------------------------------

    def _begin_key_drag(self, pt: QPoint):
        cr = self._current_crop()
        if cr is None:
            return
        self._drag_kind = "key"
        self._drag_origin_widget = pt
        self._drag_pushed_undo = False
        # Hold the actual Keyframe objects so the live mutation tracks
        # them as their source_frame changes (no lookup-by-frame).
        self._drag_keys = []
        r = self._plot_rect()
        self._drag_value_per_px = {}
        for axis, sf in self._selection:
            track = self._track_for(cr, axis)
            k = track.find_key(sf)
            if k is None:
                continue
            self._drag_keys.append((axis, track, k, sf, k.value))
            if axis not in self._drag_value_per_px:
                lo, hi = self._track_value_range(cr, axis)
                self._drag_value_per_px[axis] = (
                    (hi - lo) / max(1, r.height()))

    def _do_key_drag(self, pt: QPoint):
        if not self._drag_keys:
            return
        cr = self._current_crop()
        if cr is None:
            return
        r = self._plot_rect()
        if r.width() <= 0 or r.height() <= 0:
            return
        d_frame = ((pt.x() - self._drag_origin_widget.x())
                   / r.width() * (self._x_max - self._x_min))
        dy_px = pt.y() - self._drag_origin_widget.y()
        self._ensure_drag_undo()
        new_sel: List[Tuple[str, int]] = []
        touched_tracks = []
        for axis, track, k, sf0, v0 in self._drag_keys:
            value_per_px = self._drag_value_per_px.get(axis, 1.0)
            k.source_frame = max(0, int(round(sf0 + d_frame)))
            k.value = v0 - dy_px * value_per_px  # Y is flipped
            new_sel.append((axis, k.source_frame))
            if track not in touched_tracks:
                touched_tracks.append(track)
        # Keep each track sorted so sample()/curve rendering stay valid.
        for track in touched_tracks:
            track.keys.sort(key=lambda kf: kf.source_frame)
        self._selection = new_sel
        self.update()
        self.live_edit_changed.emit()

    def _commit_key_drag(self):
        if not self._drag_keys or not self._drag_pushed_undo:
            return
        # Resolve any frame collisions left by the drag: one key per
        # frame, the dragged key winning. Identity by id() because
        # Keyframe is an unhashable dataclass.
        per_track = {}
        for _axis, track, k, _sf0, _v0 in self._drag_keys:
            per_track.setdefault(id(track), [track, set()])[1].add(id(k))
        for track, dragged_ids in per_track.values():
            seen = {}
            for kf in track.keys:           # non-dragged first
                if id(kf) not in dragged_ids:
                    seen[kf.source_frame] = kf
            for kf in track.keys:           # dragged win on collision
                if id(kf) in dragged_ids:
                    seen[kf.source_frame] = kf
            track.keys = sorted(seen.values(),
                                key=lambda kf: kf.source_frame)
        self._selection = [(axis, k.source_frame)
                           for axis, _t, k, _s, _v in self._drag_keys]
        self._timeline.clips_changed.emit()

    def _do_handle_drag(self, pt: QPoint):
        if self._drag_handle is None:
            return
        cr = self._current_crop()
        if cr is None:
            return
        axis, _track, k, side, start_in, start_out = self._drag_handle
        # Work in pixel space so the two handles read as one straight
        # line on screen (frames vs value have different pixel scales).
        kx = self._frame_to_px(k.source_frame)
        ky = self._value_to_px(cr, axis, k.value)
        dpx = pt.x() - kx
        dpy = pt.y() - ky
        # Clamp the dragged handle to its side (in points back in time,
        # out points forward) so the curve's x stays monotonic.
        if side == "in" and dpx > 0:
            dpx = 0.0
        if side == "out" and dpx < 0:
            dpx = 0.0

        def _px_to_handle(px: float, py: float):
            return (self._px_to_frame(kx + px) - k.source_frame,
                    self._px_to_value(cr, axis, ky + py) - k.value)

        self._ensure_drag_undo()
        dragged = _px_to_handle(dpx, dpy)
        # Opposite handle: keep its captured start length, rotate to the
        # exact opposite screen direction (linked "smooth" handle).
        opp_start = start_in if side == "out" else start_out
        opp_px0 = self._frame_to_px(k.source_frame + opp_start[0]) - kx
        opp_py0 = self._value_to_px(cr, axis, k.value + opp_start[1]) - ky
        opp_len = math.hypot(opp_px0, opp_py0)
        drag_len = math.hypot(dpx, dpy)
        opposite = None
        if opp_len > 1e-6 and drag_len > 1e-6:
            ux, uy = -dpx / drag_len, -dpy / drag_len
            opposite = _px_to_handle(ux * opp_len, uy * opp_len)

        if side == "in":
            k.in_handle = dragged
            if opposite is not None:
                k.out_handle = opposite
        else:
            k.out_handle = dragged
            if opposite is not None:
                k.in_handle = opposite
        k.interp = INTERP_BEZIER
        self.update()
        self.live_edit_changed.emit()

    def _commit_handle_drag(self):
        if self._drag_handle is None or not self._drag_pushed_undo:
            return
        self._timeline.clips_changed.emit()

    def _ensure_drag_undo(self):
        """Push exactly one undo snapshot per drag, lazily on the first
        effective change (so a pure click that starts no real edit
        doesn't pollute the undo stack)."""
        if not self._drag_pushed_undo:
            self._timeline._push_undo()  # noqa: SLF001 — intentional
            self._drag_pushed_undo = True

    def _cancel_drag(self):
        self._drag_kind = ""
        self._drag_keys = []
        self._drag_handle = None
        self._drag_pushed_undo = False

    # --- Add / interp / delete -----------------------------------------

    def _add_key_at(self, axis: str, pt: QPoint):
        cr = self._current_crop()
        if cr is None:
            return
        frame = int(round(self._px_to_frame(pt.x())))
        if self._track_for(cr, axis).has_key_at(frame):
            return
        # Seed value = current curve sample so the keyframe lands on
        # the visible line.
        v = self._track_for(cr, axis).sample(frame)
        if v is None:
            v = float(cr.base_value(axis))
        self._timeline.set_crop_keyframe_interp  # ensure attribute exists
        # No public "create key" helper on the model — the underlying
        # toggle/setters either toggle or paired-axis. Mutate the track
        # directly through the model's undo path via move_crop_keyframe
        # would require an existing key. Use a dedicated path:
        ok = self._add_single_key(axis, frame, float(v))
        if ok:
            self._selection = [(axis, frame)]
            self.update()

    def _add_single_key(self, axis: str, frame: int, value: float) -> bool:
        """Add a single keyframe via a manual snapshot + direct track
        mutation. The model's higher-level helpers either toggle-pair
        (Position/Size) or require an existing key (move/delete) —
        single-axis adds from the graph editor fall through this path
        so the undo stack still wraps the operation."""
        cr = self._current_crop()
        if cr is None:
            return False
        self._timeline._push_undo()  # noqa: SLF001 — intentional
        cr.track_for(axis).set_key(frame, value, INTERP_LINEAR)
        self._timeline.clips_changed.emit()
        return True

    def _delete_selected_keys(self):
        if not self._selection:
            return
        for axis, sf in list(self._selection):
            self._timeline.delete_crop_keyframe(
                self._clip_id, self._crop_id, axis, sf)
        self._selection.clear()
        self.update()

    def _set_selected_interp(self, interp: str):
        if not self._selection:
            return
        cr = self._current_crop()
        if cr is None:
            return
        for axis, sf in self._selection:
            k = self._track_for(cr, axis).find_key(sf)
            if k is None:
                continue
            # Seed bezier handles with reasonable defaults so the user
            # has something to grab. Default: 1/3 of the segment span
            # on each side, value-flat (dy=0).
            in_handle = k.in_handle
            out_handle = k.out_handle
            if interp == INTERP_BEZIER and (in_handle == (0.0, 0.0)
                                            and out_handle == (0.0, 0.0)):
                # Pick a span based on neighbour keys (or 10 frames).
                track = self._track_for(cr, axis)
                prev_f = track.prev_key_frame(sf)
                next_f = track.next_key_frame(sf)
                in_span = (sf - prev_f) / 3.0 if prev_f is not None else 10.0
                out_span = (next_f - sf) / 3.0 if next_f is not None else 10.0
                in_handle = (-in_span, 0.0)
                out_handle = (out_span, 0.0)
            self._timeline.set_crop_keyframe_interp(
                self._clip_id, self._crop_id, axis, sf,
                interp, in_handle, out_handle)
        self.update()


class KeyframeEditorDock(QDockWidget):
    """Top-level dock that wraps ``KeyframeGraph`` plus a small toolbar.

    Defaults to floating (own top-level window) but can be docked into
    the MainWindow's right or bottom dock area. MainWindow holds the
    instance and calls ``show_for_crop()`` from the clip-info panel's
    "Edit curves…" signal."""

    playhead_scrub_requested = Signal(int)
    live_edit_changed = Signal()

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__("Keyframe Editor", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        self._timeline = timeline
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 6)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel("No crop")
        self._title_label.setStyleSheet(
            "color: #ddd; font-weight: bold;")
        toolbar.addWidget(self._title_label)
        toolbar.addStretch(1)
        self._axis_boxes: Dict[str, QCheckBox] = {}
        for axis in _AXES:
            box = QCheckBox(_TRACK_COLORS[axis][1])
            box.setChecked(True)
            box.setStyleSheet(
                f"QCheckBox {{ color: {_TRACK_COLORS[axis][0]};"
                " font-weight: bold; }")
            box.toggled.connect(
                lambda checked, a=axis:
                self._graph.set_track_visible(a, checked))
            toolbar.addWidget(box)
            self._axis_boxes[axis] = box
        fit_btn = QPushButton("Fit")
        fit_btn.setFixedWidth(40)
        fit_btn.clicked.connect(lambda: self._graph.fit_x_axis())
        toolbar.addWidget(fit_btn)
        layout.addLayout(toolbar)

        # Graph
        self._graph = KeyframeGraph(timeline, container)
        self._graph.playhead_scrub_requested.connect(
            self.playhead_scrub_requested)
        self._graph.live_edit_changed.connect(self.live_edit_changed)
        layout.addWidget(self._graph, 1)

        self.setWidget(container)

    def show_for_crop(self, clip_id: str, crop_id: str,
                      crop_label: str = "", src_fps: float = 0.0):
        self._graph.set_target(clip_id, crop_id, src_fps)
        title = "Keyframe Editor"
        if crop_label:
            title += f" — {crop_label}"
        else:
            title += f" — {crop_id[:6]}"
        self.setWindowTitle(title)
        self._title_label.setText(crop_label or crop_id[:6])
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def clear_target(self):
        self._graph.set_target(None, None)
        self._title_label.setText("No crop")
        self.setWindowTitle("Keyframe Editor")

    def set_playhead(self, source_frame: int):
        self._graph.set_playhead(source_frame)

    def current_clip_id(self) -> Optional[str]:
        return self._graph._clip_id

    def current_crop_id(self) -> Optional[str]:
        return self._graph._crop_id
