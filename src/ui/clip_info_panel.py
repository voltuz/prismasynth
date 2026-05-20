"""Right-column "Clip" panel.

Top half: read-only clip details (source / in / out / duration / dims / fps /
audio).

Bottom half: per-clip **Cropping Regions** — checkable "Edit crops in
preview" toggle, "+ Add crop" button, and a scrollable list of rows
(eye-active / label / aspect / group / trash) backed by
``Clip.crop_regions``. The window position itself is dragged on the
timeline strip; this panel no longer surfaces an "anchor frame"
control.
"""

from typing import Dict, Optional

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIntValidator, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
    QWidget,
)

from core.clip import Clip
from core.crop_region import (
    ASPECT_PRESETS, CropRegion, Segment, can_host_crop, clamp_anchor,
    required_source_frames, resolve_aspect,
)
from core.timeline import TimelineModel
from core.ui_scale import ui_scale
from core.video_source import VideoSource
from ui.icon_loader import icon


# Order shown in the aspect combo. "custom" is rendered last; selecting it
# opens a W:H dialog.
_ASPECT_ORDER = ["free", "1:1", "4:3", "16:9", "9:16", "3:4", "custom"]


def _aspect_display(aspect: str, cw: int, ch: int) -> str:
    if aspect == "custom" and cw > 0 and ch > 0:
        return f"Custom ({cw}:{ch})"
    if aspect == "custom":
        return "Custom…"
    return aspect


class _CustomAspectDialog(QDialog):
    """Tiny W:H prompt for custom aspect-ratio entry."""

    def __init__(self, initial_w: int, initial_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Aspect Ratio")
        layout = QVBoxLayout(self)
        form = QHBoxLayout()
        self._w = QSpinBox()
        self._w.setRange(1, 9999)
        self._w.setValue(max(1, initial_w or 1))
        self._h = QSpinBox()
        self._h.setRange(1, 9999)
        self._h.setValue(max(1, initial_h or 1))
        form.addWidget(QLabel("W:"))
        form.addWidget(self._w)
        form.addWidget(QLabel("H:"))
        form.addWidget(self._h)
        layout.addLayout(form)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def values(self) -> tuple:
        return (self._w.value(), self._h.value())


class _KeyframeDiamond(QAbstractButton):
    """Custom-painted AE/Premiere-style keyframe diamond toggle.

    Three states drive the fill:
      - ``"empty"`` — no keys on the group's tracks at all → dim hollow.
      - ``"half"``  — track has keys but none at the playhead → hollow accent.
      - ``"on"``    — a key exists at the playhead → solid accent.

    Clicking toggles the key at the playhead (owner wires ``clicked``).
    ``set_state`` early-outs on no-change so playback doesn't churn
    repaints (the v0.17.1 anti-stutter discipline)."""

    _DIM = QColor("#666666")
    _ACCENT = QColor("#e8a735")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "empty"
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        s = ui_scale()
        self.setFixedSize(s.px(22), s.px(22))

    def state(self) -> str:
        return self._state

    def set_state(self, state: str):
        if state == self._state:
            return
        self._state = state
        self.update()

    def sizeHint(self) -> QSize:
        s = ui_scale()
        return QSize(s.px(22), s.px(22))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        cx = rect.center().x() + 0.5
        cy = rect.center().y() + 0.5
        r = min(rect.width(), rect.height()) / 2.0 - ui_scale().px(4)
        diamond = QPolygonF([
            QPointF(cx, cy - r),
            QPointF(cx + r, cy),
            QPointF(cx, cy + r),
            QPointF(cx - r, cy),
        ])
        outline = max(1, ui_scale().px(1.5))
        if self._state == "on":
            p.setPen(QPen(self._ACCENT, 1))
            p.setBrush(QBrush(self._ACCENT))
        elif self._state == "half":
            p.setPen(QPen(self._ACCENT, outline))
            p.setBrush(Qt.BrushStyle.NoBrush)
        else:  # empty
            p.setPen(QPen(self._DIM, outline))
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(diamond)
        p.end()


class _CropRow(QFrame):
    """One row in the cropping list: eye / label / aspect / group /
    trash. Keyframe controls live in a dedicated section below the crop
    list (see ``ClipInfoPanel`` Keyframes group), not in the row."""

    active_toggled = Signal(str)
    label_changed = Signal(str, str)
    aspect_changed = Signal(str, str, int, int)       # crop_id, preset, cw, ch
    group_changed = Signal(str, object)               # crop_id, group_id or None
    delete_requested = Signal(str)
    selected = Signal(str)

    def __init__(self, crop: CropRegion, groups: Dict[str, "Group"],
                 parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_CropRow { border: 1px solid #3c3c3c; border-radius: 3px; }"
            "_CropRow[selected=\"true\"] { border: 1px solid #5577aa; }"
        )
        self._crop_id = crop.id
        self._suppress = False

        s = ui_scale()
        layout = QGridLayout(self)
        layout.setContentsMargins(s.px(6), s.px(4), s.px(6), s.px(4))
        layout.setHorizontalSpacing(s.px(6))
        layout.setVerticalSpacing(s.px(2))

        # Row 0: eye + label + trash
        self._eye_btn = QToolButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._eye_btn.setFixedSize(s.px(22), s.px(22))
        self._eye_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._eye_icon_on = icon("eye")
        self._eye_icon_off = icon("eye-off")
        self._eye_btn.setToolTip("Active — toggle to skip on export")
        self._eye_btn.toggled.connect(self._on_eye_toggled)
        layout.addWidget(self._eye_btn, 0, 0)

        self._label_edit = QLineEdit(crop.label)
        self._label_edit.setPlaceholderText(f"Crop {crop.id[:6]}")
        self._label_edit.editingFinished.connect(self._on_label_committed)
        layout.addWidget(self._label_edit, 0, 1, 1, 4)

        self._delete_btn = QToolButton()
        self._delete_btn.setIcon(icon("trash"))
        self._delete_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._delete_btn.setFixedSize(s.px(22), s.px(22))
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Remove crop")
        self._delete_btn.setStyleSheet(
            "QToolButton { background: transparent;"
            " border: 1px solid #555; border-radius: 3px; }"
            "QToolButton:hover { background: rgba(232, 122, 117, 40); }"
        )
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._crop_id))
        layout.addWidget(self._delete_btn, 0, 5)

        # Row 1: aspect + group
        layout.addWidget(QLabel("Aspect:"), 1, 0, 1, 1,
                         Qt.AlignmentFlag.AlignRight)
        self._aspect_combo = QComboBox()
        for key in _ASPECT_ORDER:
            self._aspect_combo.addItem(_aspect_display(key, 0, 0), key)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        layout.addWidget(self._aspect_combo, 1, 1, 1, 2)

        layout.addWidget(QLabel("Group:"), 1, 3, 1, 1,
                         Qt.AlignmentFlag.AlignRight)
        self._group_combo = QComboBox()
        self._group_combo.currentIndexChanged.connect(self._on_group_changed)
        layout.addWidget(self._group_combo, 1, 4, 1, 2)

        self.update_from(crop, groups)

    # ------------------------------------------------------------------

    def crop_id(self) -> str:
        return self._crop_id

    def set_selected(self, selected: bool):
        self.setProperty("selected", "true" if selected else "false")
        # Stylesheet refresh after a dynamic property change.
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self._crop_id)
            # Accept so the click doesn't bubble to the rows host, whose
            # blank-space event filter would otherwise immediately
            # deselect the crop we just selected.
            event.accept()
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------

    def update_from(self, crop: CropRegion, groups: Dict[str, "Group"]):
        self._suppress = True
        try:
            self._eye_btn.setChecked(crop.active)
            self._eye_btn.setIcon(
                self._eye_icon_on if crop.active else self._eye_icon_off)
            self._eye_btn.setToolTip(
                "Active — toggle to skip on export" if crop.active
                else "Inactive — toggle to include on export")
            if self._label_edit.text() != crop.label:
                self._label_edit.setText(crop.label)
            self._label_edit.setPlaceholderText(f"Crop {crop.id[:6]}")

            # Aspect — refresh the "Custom" label so it shows the W:H.
            idx = self._aspect_combo.findData(crop.aspect_ratio)
            if idx < 0:
                idx = self._aspect_combo.findData("free")
            self._aspect_combo.setItemText(
                self._aspect_combo.findData("custom"),
                _aspect_display("custom", crop.custom_ratio_w,
                                crop.custom_ratio_h))
            self._aspect_combo.setCurrentIndex(idx)

            # Group combo: rebuild with current groups.
            self._group_combo.clear()
            self._group_combo.addItem("(Untagged)", None)
            for gid, g in groups.items():
                self._group_combo.addItem(g.name, gid)
            target = self._group_combo.findData(crop.group_id)
            if target < 0:
                target = 0
            self._group_combo.setCurrentIndex(target)
        finally:
            self._suppress = False

    # --- Slots --------------------------------------------------------

    def _on_eye_toggled(self, _checked: bool):
        if self._suppress:
            return
        self.active_toggled.emit(self._crop_id)

    def _on_label_committed(self):
        if self._suppress:
            return
        self.label_changed.emit(self._crop_id, self._label_edit.text())

    def _on_aspect_changed(self, idx: int):
        if self._suppress or idx < 0:
            return
        key = self._aspect_combo.itemData(idx)
        if key == "custom":
            dlg = _CustomAspectDialog(0, 0, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                cw, ch = dlg.values()
                self.aspect_changed.emit(self._crop_id, "custom", cw, ch)
            else:
                # User cancelled — revert combo to "free" so we don't sit
                # in a halfway state.
                self._suppress = True
                try:
                    self._aspect_combo.setCurrentIndex(
                        self._aspect_combo.findData("free"))
                finally:
                    self._suppress = False
                self.aspect_changed.emit(self._crop_id, "free", 0, 0)
            return
        self.aspect_changed.emit(self._crop_id, key, 0, 0)

    def _on_group_changed(self, idx: int):
        if self._suppress or idx < 0:
            return
        data = self._group_combo.itemData(idx)
        self.group_changed.emit(self._crop_id, data)


class _SegmentRow(QFrame):
    """One export-segment row: active eye + '@ frame N' + trash."""

    active_toggled = Signal(str)        # segment_id
    delete_requested = Signal(str)      # segment_id
    selected = Signal(str)              # segment_id

    def __init__(self, segment: Segment, can_delete: bool, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_SegmentRow { border: 1px solid #3c3c3c; border-radius: 3px; }"
            "_SegmentRow[selected=\"true\"] { border: 1px solid #5577aa; }"
        )
        self._segment_id = segment.id
        s = ui_scale()
        row = QHBoxLayout(self)
        row.setContentsMargins(s.px(6), s.px(3), s.px(6), s.px(3))
        row.setSpacing(s.px(6))

        self._eye_btn = QToolButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._eye_btn.setFixedSize(s.px(22), s.px(22))
        self._eye_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._eye_icon_on = icon("eye")
        self._eye_icon_off = icon("eye-off")
        self._eye_btn.toggled.connect(self._on_eye)
        row.addWidget(self._eye_btn)

        self._frame_label = QLabel()
        self._frame_label.setStyleSheet("color: #ccc;")
        row.addWidget(self._frame_label, 1)

        self._delete_btn = QToolButton()
        self._delete_btn.setIcon(icon("trash"))
        self._delete_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._delete_btn.setFixedSize(s.px(22), s.px(22))
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Remove segment")
        self._delete_btn.setStyleSheet(
            "QToolButton { background: transparent;"
            " border: 1px solid #555; border-radius: 3px; }"
            "QToolButton:hover { background: rgba(232, 122, 117, 40); }"
        )
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._segment_id))
        row.addWidget(self._delete_btn)

        self._suppress = False
        self.update_from(segment, can_delete)

    def segment_id(self) -> str:
        return self._segment_id

    def update_from(self, segment: Segment, can_delete: bool):
        self._suppress = True
        try:
            self._eye_btn.setChecked(segment.active)
            self._eye_btn.setIcon(
                self._eye_icon_on if segment.active else self._eye_icon_off)
            self._eye_btn.setToolTip(
                "Active — toggle to skip this segment on export"
                if segment.active
                else "Inactive — toggle to include this segment on export")
            self._frame_label.setText(f"@ frame {segment.anchor_frame}")
            # The last remaining segment can't be removed.
            self._delete_btn.setEnabled(can_delete)
        finally:
            self._suppress = False

    def set_selected(self, selected: bool):
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self._segment_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def _on_eye(self, _checked: bool):
        if self._suppress:
            return
        self.active_toggled.emit(self._segment_id)


class ClipInfoPanel(QWidget):
    """Right-column "Clip" panel — details + cropping regions."""

    # Bool: True when "Edit crops in preview" is checked (and a non-gap clip
    # is selected). MainWindow forwards this to PreviewWidget.
    crop_edit_mode_changed = Signal(bool)
    # Crop selection sync: row click in the panel ⇄ click in the preview.
    # Empty string means "deselect".
    crop_selected = Signal(str)
    # Keyframe nav: user clicked a prev/next arrow. MainWindow converts
    # the source frame to a timeline frame for the clip currently
    # holding this crop and moves the playhead.
    crop_jump_to_source_frame = Signal(str, str, int)  # clip_id, crop_id, source_frame
    # User clicked "Edit curves…" on a row. MainWindow opens / focuses
    # the keyframe-editor dock and points it at this crop.
    crop_edit_curves_requested = Signal(str, str)      # clip_id, crop_id
    # User selected an export segment (panel side). MainWindow mirrors
    # the halo onto the timeline.
    crop_segment_selected = Signal(str, str, str)      # clip_id, crop_id, segment_id

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources: Dict[str, VideoSource] = {}
        self._current_clip: Optional[Clip] = None
        self._current_source: Optional[VideoSource] = None
        self._crop_rows: Dict[str, _CropRow] = {}
        self._selected_crop_id: Optional[str] = None
        self._selected_segment_id: Optional[str] = None
        self._seg_rows: Dict[str, "_SegmentRow"] = {}
        # Playhead source frame is set by MainWindow on every playhead change
        # so "From playhead" can fill the anchor without a round-trip.
        self._playhead_source_frame: int = 0
        # The crop the dedicated Keyframes section currently targets, and
        # cached jump targets (group, direction) -> source frame | None
        # populated by ``_refresh_keyframe_section`` for the prev/next
        # nav arrows.
        self._kf_section_crop_id: Optional[str] = None
        self._kf_jump_targets: Dict[tuple, Optional[int]] = {}
        self._kf_suppress: bool = False

        s = ui_scale()
        self.setMinimumWidth(s.px(220))
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(s.px(8), s.px(8), s.px(8), s.px(8))
        self._outer_layout = layout
        ui_scale().changed.connect(self._on_ui_scale_changed)

        self._title = QLabel("No clip selected")
        self._title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #ddd;")
        layout.addWidget(self._title)

        # ---- Clip Details ------------------------------------------------
        self._group = QGroupBox("Clip Details")
        form = QFormLayout()
        self._source_label = QLabel("-")
        self._in_label = QLabel("-")
        self._out_label = QLabel("-")
        self._duration_label = QLabel("-")
        self._resolution_label = QLabel("-")
        self._fps_label = QLabel("-")
        self._audio_label = QLabel("-")
        for lbl in [self._source_label, self._in_label, self._out_label,
                    self._duration_label, self._resolution_label,
                    self._fps_label, self._audio_label]:
            lbl.setStyleSheet("color: #ccc;")
        form.addRow("Source:", self._source_label)
        form.addRow("In:", self._in_label)
        form.addRow("Out:", self._out_label)
        form.addRow("Duration:", self._duration_label)
        form.addRow("Resolution:", self._resolution_label)
        form.addRow("FPS:", self._fps_label)
        form.addRow("Audio:", self._audio_label)
        self._group.setLayout(form)
        layout.addWidget(self._group)

        # ---- Cropping Regions -------------------------------------------
        self._crops_group = QGroupBox("Cropping Regions")
        crops_layout = QVBoxLayout(self._crops_group)
        crops_layout.setContentsMargins(s.px(6), s.px(6), s.px(6), s.px(6))
        crops_layout.setSpacing(s.px(6))

        # Top row: edit-mode toggle + add button
        controls = QHBoxLayout()
        self._edit_btn = QPushButton("Edit crops in preview")
        self._edit_btn.setCheckable(True)
        self._edit_btn.toggled.connect(self._on_edit_toggled)
        controls.addWidget(self._edit_btn, 1)
        self._add_btn = QPushButton("+ Add crop")
        self._add_btn.clicked.connect(self._on_add_crop)
        controls.addWidget(self._add_btn)
        crops_layout.addLayout(controls)

        # Inline status hint (clip too short / no clip / gap selected)
        self._hint_label = QLabel()
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color: #888; font-size: 11px;")
        crops_layout.addWidget(self._hint_label)

        # Scrollable rows area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(s.px(4))
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_host)
        crops_layout.addWidget(self._scroll, 1)

        layout.addWidget(self._crops_group, 1)

        # ---- Segments (export windows of the selected crop) ------------
        self._build_segments_section(layout)

        # ---- Keyframes (dedicated section for the selected crop) --------
        self._build_keyframe_section(layout)

        # Blank-space deselect: filter mouse presses delivered to the
        # Clip-panel container backgrounds. installEventFilter only sees
        # events delivered to that exact widget (not its children), so
        # each fires only when the click lands on its own blank surface
        # — interactive children that consume their clicks never trigger
        # a deselect.
        for w in (self, self._group, self._crops_group, self._kf_group,
                  self._scroll.viewport(), self._rows_host):
            w.installEventFilter(self)

        # Refresh hooks — clips_changed covers crop add/remove/edit; we also
        # refresh on groups_changed so the group combos in each row stay
        # in sync with the registry.
        self._timeline.clips_changed.connect(self._refresh_crops)
        self._timeline.groups_changed.connect(self._refresh_crops)

        self._update_crop_section_enabled()
        self._refresh_keyframe_section()
        self._refresh_segments_section()

    def eventFilter(self, obj, event):
        # Click on blank Clip-panel space → clear crop selection.
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and self._selected_crop_id is not None):
            self._deselect_crop()
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Keyframes section (dedicated, targets the selected crop)
    # ------------------------------------------------------------------

    def _build_keyframe_section(self, outer_layout: QVBoxLayout):
        """Build the spacious 'Keyframes' group box under the crop list.

        Shows controls for the currently-targeted crop only: a diamond
        toggle + nav + count + quick-interp per parameter group
        (Position = x,y / Size = w,h), plus an Open-graph-editor button.
        Curve/handle visualization stays in the detached editor."""
        s = ui_scale()
        self._kf_group = QGroupBox("Keyframes")
        outer = QVBoxLayout(self._kf_group)
        outer.setContentsMargins(s.px(8), s.px(8), s.px(8), s.px(8))
        outer.setSpacing(s.px(6))

        # Header: which crop + live playhead frame.
        self._kf_header = QLabel("Select a crop")
        self._kf_header.setStyleSheet("color: #ddd; font-weight: bold;")
        self._kf_header.setWordWrap(True)
        outer.addWidget(self._kf_header)

        self._kf_frame_label = QLabel("frame —")
        self._kf_frame_label.setStyleSheet("color: #888; font-size: 11px;")
        outer.addWidget(self._kf_frame_label)

        # Per-group grid. Columns: name (flex) | nav cluster | count | interp.
        grid = QGridLayout()
        grid.setHorizontalSpacing(s.px(6))
        grid.setVerticalSpacing(s.px(6))
        grid.setColumnStretch(0, 1)          # name column flexes
        self._kf_widgets: Dict[str, dict] = {}
        for row, (group, label) in enumerate(
                (("position", "Position"), ("size", "Size"))):
            name = QLabel(label)
            name.setStyleSheet("color: #ccc;")
            grid.addWidget(name, row, 0)

            # NLE-style keyframe navigator: ◀ ◆ ▶ as one tight cluster
            # (prev key · toggle key at playhead · next key).
            prev_btn = QToolButton()
            prev_btn.setText("◀")
            prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            prev_btn.setToolTip(f"Previous {label} keyframe")
            prev_btn.clicked.connect(
                lambda _=False, g=group: self._on_kf_jump(g, -1))

            diamond = _KeyframeDiamond()
            diamond.setToolTip(
                f"Toggle {label} keyframe at the playhead")
            diamond.clicked.connect(
                lambda _=False, g=group: self._on_kf_toggle(g))

            next_btn = QToolButton()
            next_btn.setText("▶")
            next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            next_btn.setToolTip(f"Next {label} keyframe")
            next_btn.clicked.connect(
                lambda _=False, g=group: self._on_kf_jump(g, +1))

            nav = QHBoxLayout()
            nav.setContentsMargins(0, 0, 0, 0)
            nav.setSpacing(s.px(2))
            nav.addWidget(prev_btn)
            nav.addWidget(diamond)
            nav.addWidget(next_btn)
            grid.addLayout(nav, row, 1)

            count = QLabel("0 keys")
            count.setStyleSheet("color: #888; font-size: 11px;")
            count.setMinimumWidth(s.px(48))
            grid.addWidget(count, row, 2)

            interp = QComboBox()
            interp.addItem("Linear", "linear")
            interp.addItem("Bezier", "bezier")
            interp.addItem("Step", "step")
            interp.setToolTip(
                f"Interpolation of the {label} keyframe at the playhead")
            interp.currentIndexChanged.connect(
                lambda _idx, g=group: self._on_kf_interp_changed(g))
            grid.addWidget(interp, row, 3)

            self._kf_widgets[group] = {
                "diamond": diamond, "name": name,
                "prev": prev_btn, "next": next_btn,
                "count": count, "interp": interp,
            }
        outer.addLayout(grid)

        self._kf_open_editor = QPushButton("Open graph editor")
        self._kf_open_editor.setCursor(Qt.CursorShape.PointingHandCursor)
        self._kf_open_editor.clicked.connect(self._on_kf_open_editor)
        outer.addWidget(self._kf_open_editor)

        outer_layout.addWidget(self._kf_group)

    # ------------------------------------------------------------------
    # Segments section (export windows of the selected crop)
    # ------------------------------------------------------------------

    def _build_segments_section(self, outer_layout: QVBoxLayout):
        s = ui_scale()
        self._seg_group = QGroupBox("Segments")
        v = QVBoxLayout(self._seg_group)
        v.setContentsMargins(s.px(8), s.px(6), s.px(8), s.px(8))
        v.setSpacing(s.px(4))

        self._seg_add_btn = QPushButton("+ Add segment")
        self._seg_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._seg_add_btn.clicked.connect(self._on_add_segment)
        v.addWidget(self._seg_add_btn)

        self._seg_hint = QLabel("Select a crop to manage export segments.")
        self._seg_hint.setWordWrap(True)
        self._seg_hint.setStyleSheet("color: #888; font-size: 11px;")
        v.addWidget(self._seg_hint)

        self._seg_scroll = QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        self._seg_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._seg_scroll.setMaximumHeight(s.px(140))
        self._seg_rows_host = QWidget()
        self._seg_rows_layout = QVBoxLayout(self._seg_rows_host)
        self._seg_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._seg_rows_layout.setSpacing(s.px(3))
        self._seg_rows_layout.addStretch(1)
        self._seg_scroll.setWidget(self._seg_rows_host)
        v.addWidget(self._seg_scroll)

        outer_layout.addWidget(self._seg_group)

    def set_selected_segment(self, segment_id: str):
        """External sync (timeline click). Highlights the row; no emit."""
        self._apply_segment_selection(segment_id, emit=False)

    def _apply_segment_selection(self, segment_id, emit: bool = True):
        """Single point of truth for export-segment selection: updates
        state, row halos, and (when ``emit``) the ``crop_segment_selected``
        signal. No-ops when unchanged."""
        sid = segment_id or None
        if sid == self._selected_segment_id:
            return
        self._selected_segment_id = sid
        for rid, row in self._seg_rows.items():
            row.set_selected(rid == sid)
        if emit:
            cr = self._kf_target_crop()
            clip_id = (self._current_clip.id
                       if (self._current_clip
                           and not self._current_clip.is_gap) else "")
            crop_id = cr.id if cr is not None else ""
            self.crop_segment_selected.emit(clip_id, crop_id, sid or "")

    def _refresh_segments_section(self):
        """Rebuild the Segments list for the targeted crop. Runs on
        selection / clips_changed, NOT on playhead moves (segment
        anchors don't depend on the playhead)."""
        # Drop old rows.
        for row in list(self._seg_rows.values()):
            self._seg_rows_layout.removeWidget(row)
            row.deleteLater()
        self._seg_rows.clear()

        cr = self._kf_target_crop()
        if cr is None:
            self._seg_hint.setVisible(True)
            self._seg_add_btn.setEnabled(False)
            self._selected_segment_id = None
            return
        self._seg_hint.setVisible(False)
        self._seg_add_btn.setEnabled(True)
        # Drop a stale selection that doesn't belong to this crop.
        if (self._selected_segment_id is not None
                and cr.find_segment(self._selected_segment_id) is None):
            self._selected_segment_id = None
        can_delete = len(cr.segments) > 1
        for seg in cr.segments:
            row = _SegmentRow(seg, can_delete, parent=self._seg_rows_host)
            row.active_toggled.connect(self._on_segment_active_toggled)
            row.delete_requested.connect(self._on_segment_delete)
            row.selected.connect(self._on_segment_row_selected)
            idx = self._seg_rows_layout.count() - 1
            self._seg_rows_layout.insertWidget(idx, row)
            self._seg_rows[seg.id] = row
            row.set_selected(seg.id == self._selected_segment_id)

    # --- Segment handlers ---------------------------------------------

    def _on_add_segment(self):
        cr = self._kf_target_crop()
        clip = self._current_clip
        source = self._current_source
        if (cr is None or clip is None or clip.is_gap
                or source is None or source.fps <= 0):
            return
        anchor = clamp_anchor(
            self._playhead_source_frame, clip, source.fps)
        seg_id = self._timeline.add_crop_segment(clip.id, cr.id, anchor)
        if seg_id:
            self._selected_segment_id = seg_id
            self.crop_segment_selected.emit(clip.id, cr.id, seg_id)

    def _on_segment_active_toggled(self, segment_id: str):
        cr = self._kf_target_crop()
        if cr is None or self._current_clip is None:
            return
        self._timeline.toggle_crop_segment_active(
            self._current_clip.id, cr.id, segment_id)

    def _on_segment_delete(self, segment_id: str):
        cr = self._kf_target_crop()
        if cr is None or self._current_clip is None:
            return
        if self._selected_segment_id == segment_id:
            self._selected_segment_id = None
        self._timeline.remove_crop_segment(
            self._current_clip.id, cr.id, segment_id)

    def _on_segment_row_selected(self, segment_id: str):
        cr = self._kf_target_crop()
        if cr is None or self._current_clip is None:
            return
        if segment_id == self._selected_segment_id:
            # Re-click the selected segment → deselect (crop stays).
            self._apply_segment_selection(None)
            return
        # Promote the sole-crop fallback to an explicit crop selection
        # (idempotent when the crop is already selected), then select.
        self._apply_crop_selection(cr.id)
        self._apply_segment_selection(segment_id)

    # ------------------------------------------------------------------

    def _on_ui_scale_changed(self):
        s = ui_scale()
        self.setMinimumWidth(s.px(220))
        self._outer_layout.setContentsMargins(
            s.px(8), s.px(8), s.px(8), s.px(8))
        self._rows_layout.setSpacing(s.px(4))

    # ------------------------------------------------------------------
    # Plumbing from MainWindow
    # ------------------------------------------------------------------

    def set_playhead_source_frame(self, frame: int):
        """MainWindow calls this on every playhead change. Same-frame
        ticks early-out — the playback timer fires at 60Hz but source
        frames advance at the source FPS (24/25/30), so most ticks
        carry the same frame value and would re-walk the KF state for
        nothing."""
        f = int(frame)
        if f == self._playhead_source_frame:
            return
        self._playhead_source_frame = f
        self._refresh_keyframe_section()

    def update_clip(self, clip: Optional[Clip],
                    sources: Dict[str, VideoSource]):
        self._sources = sources

        # If the user switches clip, drop the "edit crops" mode so the
        # overlay doesn't follow them between clips with stale state,
        # and clear the cached crop selection so the preview overlay's
        # next selection isn't aimed at a now-foreign crop_id.
        prev_clip_id = self._current_clip.id if self._current_clip else None
        new_clip_id = clip.id if clip else None
        if prev_clip_id != new_clip_id:
            if self._edit_btn.isChecked():
                self._edit_btn.setChecked(False)  # fires _on_edit_toggled
            if self._selected_crop_id is not None:
                self._selected_crop_id = None
                self.crop_selected.emit("")

        self._current_clip = clip
        self._current_source = (sources.get(clip.source_id)
                                if (clip and not clip.is_gap) else None)

        # Clip Details refresh — same logic as before.
        if clip is None:
            self._title.setText("No clip selected")
            for lbl in [self._source_label, self._in_label, self._out_label,
                        self._duration_label, self._resolution_label,
                        self._fps_label, self._audio_label]:
                lbl.setText("-")
                lbl.setStyleSheet("color: #ccc;")
        elif clip.is_gap:
            self._title.setText("Gap")
            self._source_label.setText("-")
            self._in_label.setText("-")
            self._out_label.setText("-")
            self._duration_label.setText(f"{clip.duration_frames} frames")
            self._resolution_label.setText("-")
            self._fps_label.setText("-")
            self._audio_label.setText("-")
            self._audio_label.setStyleSheet("color: #ccc;")
        else:
            source = self._current_source
            source_name = (source.file_path.split("\\")[-1].split("/")[-1]
                           if source else "Unknown")
            fps = source.fps if source else 24.0
            self._title.setText(f"Clip {clip.id[:8]}")
            self._source_label.setText(source_name)
            self._in_label.setText(str(clip.source_in))
            self._out_label.setText(str(clip.source_out))
            dur_frames = clip.duration_frames
            dur_secs = dur_frames / fps if fps > 0 else 0
            self._duration_label.setText(
                f"{dur_frames} frames ({dur_secs:.2f}s)")
            if source:
                self._resolution_label.setText(
                    f"{source.width}x{source.height}")
                self._fps_label.setText(f"{source.fps:.3f}")
                audio = source.format_audio()
                self._audio_label.setText(audio)
                self._audio_label.setStyleSheet(
                    "color: #e8a735;" if audio == "none" else "color: #ccc;")
            else:
                self._resolution_label.setText("-")
                self._fps_label.setText("-")
                self._audio_label.setText("-")
                self._audio_label.setStyleSheet("color: #ccc;")

        self._refresh_crops()
        self._update_crop_section_enabled()

    def set_selected_crop(self, crop_id: str):
        """External callers (preview overlay / timeline) set the selected
        crop here. ``emit=False`` keeps this the receiving end so the
        selection doesn't loop back out as a fresh ``crop_selected``."""
        self._apply_crop_selection(crop_id, emit=False)

    def _apply_crop_selection(self, crop_id, emit: bool = True):
        """Single point of truth for crop selection: updates state, row
        highlights, the ``crop_selected`` signal, and the Keyframes
        section. No-ops when the selection is unchanged."""
        if crop_id == "":
            crop_id = None
        if crop_id == self._selected_crop_id:
            return
        self._selected_crop_id = crop_id
        for cid, row in self._crop_rows.items():
            row.set_selected(cid == crop_id)
        if emit:
            self.crop_selected.emit(crop_id or "")
        # Switching / clearing the crop clears its segment selection too
        # (and emits so the timeline halo drops). Cheap no-op when no
        # segment was selected.
        self._apply_segment_selection(None)
        self._refresh_keyframe_section()
        self._refresh_segments_section()

    def _deselect_crop(self):
        self._apply_crop_selection(None)

    # ------------------------------------------------------------------
    # Crop section state
    # ------------------------------------------------------------------

    def _update_crop_section_enabled(self):
        clip = self._current_clip
        source = self._current_source
        if clip is None:
            self._hint_label.setText("Select a clip to manage crops.")
            self._edit_btn.setEnabled(False)
            self._add_btn.setEnabled(False)
            return
        if clip.is_gap:
            self._hint_label.setText("Gaps don't have crops.")
            self._edit_btn.setEnabled(False)
            self._add_btn.setEnabled(False)
            return
        fps = source.fps if source else 0.0
        if fps <= 0:
            self._hint_label.setText("Source FPS unknown — cannot crop.")
            self._edit_btn.setEnabled(False)
            self._add_btn.setEnabled(False)
            return
        req = required_source_frames(fps)
        if not can_host_crop(clip, fps):
            self._hint_label.setText(
                f"Clip too short — needs at least {req} source frames "
                f"for an 81-frame export at 16fps.")
            self._edit_btn.setEnabled(False)
            self._add_btn.setEnabled(False)
            return
        self._hint_label.setText("")
        self._edit_btn.setEnabled(True)
        self._add_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Crop row rebuild
    # ------------------------------------------------------------------

    def _refresh_crops(self):
        """Rebuild the row widgets from the current clip's crop_regions.
        Cheap and bulletproof — rebuilds on every clips_changed."""
        clip = self._current_clip
        # Drop all existing rows; we rebuild fresh each time.
        for row in list(self._crop_rows.values()):
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._crop_rows.clear()

        if clip is None or clip.is_gap or self._current_source is None:
            self._update_crop_section_enabled()
            self._refresh_keyframe_section()
            self._refresh_segments_section()
            return

        # Re-resolve in case the live model object differs from the cached
        # reference (snapshot/restore swaps instances).
        live = self._timeline.get_clip_by_id(clip.id)
        if live is None:
            self._update_crop_section_enabled()
            self._refresh_keyframe_section()
            self._refresh_segments_section()
            return
        self._current_clip = live
        clip = live

        groups = self._timeline.groups

        for cr in clip.crop_regions:
            row = _CropRow(cr, groups, parent=self._rows_host)
            row.active_toggled.connect(self._on_row_active_toggled)
            row.label_changed.connect(self._on_row_label_changed)
            row.aspect_changed.connect(self._on_row_aspect_changed)
            row.group_changed.connect(self._on_row_group_changed)
            row.delete_requested.connect(self._on_row_delete)
            row.selected.connect(self._on_row_selected)
            idx = self._rows_layout.count() - 1
            self._rows_layout.insertWidget(idx, row)
            self._crop_rows[cr.id] = row
            if cr.id == self._selected_crop_id:
                row.set_selected(True)

        self._update_crop_section_enabled()
        self._refresh_keyframe_section()
        self._refresh_segments_section()

    # ------------------------------------------------------------------
    # Keyframe state refresh (playhead-driven)
    # ------------------------------------------------------------------

    @staticmethod
    def _group_tracks(cr, group: str):
        """Return the two axis tracks for a parameter group."""
        if group == "position":
            return (cr.x_track, cr.y_track)
        return (cr.w_track, cr.h_track)

    @staticmethod
    def _combine_group_state(track_a, track_b, frame: int) -> str:
        """Diamond visual state for an axis pair: 'on' if either axis
        has a key at this frame, 'half' if either has any key elsewhere,
        else 'empty'."""
        if not track_a and not track_b:
            return "empty"
        if track_a.has_key_at(frame) or track_b.has_key_at(frame):
            return "on"
        return "half"

    @staticmethod
    def _group_prev(track_a, track_b, frame: int):
        a = track_a.prev_key_frame(frame)
        b = track_b.prev_key_frame(frame)
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)  # closest preceding key

    @staticmethod
    def _group_next(track_a, track_b, frame: int):
        a = track_a.next_key_frame(frame)
        b = track_b.next_key_frame(frame)
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)  # closest following key

    @staticmethod
    def _group_key_count(track_a, track_b) -> int:
        """Distinct keyed frames across the two axes of a group."""
        frames = {k.source_frame for k in track_a.keys}
        frames |= {k.source_frame for k in track_b.keys}
        return len(frames)

    @staticmethod
    def _group_interp_at(track_a, track_b, frame: int):
        """Common interp of the group's keys at ``frame``, or None when
        no key exists there or the two axes disagree."""
        interps = set()
        for tr in (track_a, track_b):
            k = tr.find_key(frame)
            if k is not None:
                interps.add(k.interp)
        if len(interps) == 1:
            return next(iter(interps))
        return None

    def _kf_target_crop(self):
        """Resolve which crop the Keyframes section reflects: the
        selected crop if it belongs to the current clip, else the sole
        crop when there's exactly one (display-only), else None."""
        clip = self._current_clip
        if clip is None or clip.is_gap:
            return None
        live = self._timeline.get_clip_by_id(clip.id)
        if live is None:
            return None
        crops = live.crop_regions
        if self._selected_crop_id:
            for cr in crops:
                if cr.id == self._selected_crop_id:
                    return cr
        if len(crops) == 1:
            return crops[0]
        return None

    def _refresh_keyframe_section(self):
        """Update the dedicated Keyframes section for the target crop.

        Runs on playhead changes (already same-frame-guarded upstream),
        clips_changed, and selection changes. The diamond widgets and
        combos early-out on unchanged values, so this is cheap to call
        and won't churn during playback."""
        cr = self._kf_target_crop()
        self._kf_section_crop_id = cr.id if cr is not None else None
        frame = int(self._playhead_source_frame)

        if cr is None:
            self._kf_header.setText("Select a crop")
            self._kf_frame_label.setText("frame —")
            self._kf_jump_targets = {}
            for w in self._kf_widgets.values():
                w["diamond"].set_state("empty")
                w["diamond"].setEnabled(False)
                w["prev"].setEnabled(False)
                w["next"].setEnabled(False)
                w["count"].setText("—")
                self._set_interp_combo(w["interp"], None, enabled=False)
            self._kf_open_editor.setEnabled(False)
            return

        label = cr.label or f"Crop {cr.id[:6]}"
        self._kf_header.setText(f"Keyframes — {label}")
        self._kf_frame_label.setText(f"frame {frame}")
        self._kf_open_editor.setEnabled(True)

        jump: Dict[tuple, Optional[int]] = {}
        for group in ("position", "size"):
            ta, tb = self._group_tracks(cr, group)
            w = self._kf_widgets[group]
            w["diamond"].setEnabled(True)
            w["diamond"].set_state(
                self._combine_group_state(ta, tb, frame))
            prev_f = self._group_prev(ta, tb, frame)
            next_f = self._group_next(ta, tb, frame)
            jump[(group, -1)] = prev_f
            jump[(group, +1)] = next_f
            w["prev"].setEnabled(prev_f is not None)
            w["next"].setEnabled(next_f is not None)
            n = self._group_key_count(ta, tb)
            w["count"].setText(f"{n} key" if n == 1 else f"{n} keys")
            interp = self._group_interp_at(ta, tb, frame)
            self._set_interp_combo(
                w["interp"], interp, enabled=interp is not None)
        self._kf_jump_targets = jump

    def _set_interp_combo(self, combo: QComboBox, interp,
                          enabled: bool):
        """Set the interp combo selection without firing its change
        handler. ``interp`` None leaves the current index but disables."""
        self._kf_suppress = True
        try:
            combo.setEnabled(enabled)
            if interp is not None:
                idx = combo.findData(interp)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        finally:
            self._kf_suppress = False

    # ------------------------------------------------------------------
    # Keyframes section handlers
    # ------------------------------------------------------------------

    def _on_kf_toggle(self, group: str):
        if (self._current_clip is None
                or self._kf_section_crop_id is None):
            return
        self._timeline.toggle_crop_keyframe(
            self._current_clip.id, self._kf_section_crop_id, group,
            int(self._playhead_source_frame))

    def _on_kf_jump(self, group: str, direction: int):
        if (self._current_clip is None
                or self._kf_section_crop_id is None):
            return
        target = self._kf_jump_targets.get((group, direction))
        if target is None:
            return
        self.crop_jump_to_source_frame.emit(
            self._current_clip.id, self._kf_section_crop_id, int(target))

    def _on_kf_interp_changed(self, group: str):
        if (self._kf_suppress or self._current_clip is None
                or self._kf_section_crop_id is None):
            return
        combo = self._kf_widgets[group]["interp"]
        interp = combo.currentData()
        self._timeline.set_crop_keyframe_group_interp(
            self._current_clip.id, self._kf_section_crop_id, group,
            int(self._playhead_source_frame), interp)

    def _on_kf_open_editor(self):
        if (self._current_clip is None
                or self._kf_section_crop_id is None):
            return
        self.crop_edit_curves_requested.emit(
            self._current_clip.id, self._kf_section_crop_id)

    # ------------------------------------------------------------------
    # Row callbacks → TimelineModel mutations
    # ------------------------------------------------------------------

    def _on_row_active_toggled(self, crop_id: str):
        if self._current_clip is None:
            return
        self._timeline.toggle_crop_active(self._current_clip.id, crop_id)

    def _on_row_label_changed(self, crop_id: str, label: str):
        if self._current_clip is None:
            return
        self._timeline.update_crop_region(
            self._current_clip.id, crop_id, label=label.strip())

    def _on_row_aspect_changed(self, crop_id: str, preset: str,
                               cw: int, ch: int):
        if self._current_clip is None:
            return
        self._timeline.update_crop_region(
            self._current_clip.id, crop_id,
            aspect_ratio=preset,
            custom_ratio_w=cw,
            custom_ratio_h=ch,
        )

    def _on_row_group_changed(self, crop_id: str, group_id):
        if self._current_clip is None:
            return
        self._timeline.update_crop_region(
            self._current_clip.id, crop_id, group_id=group_id)

    def _on_row_delete(self, crop_id: str):
        if self._current_clip is None:
            return
        # Confirm only when the crop has been explicitly tagged with a group
        # (i.e. the user invested something into it). Untagged blank crops
        # delete silently.
        clip = self._timeline.get_clip_by_id(self._current_clip.id)
        target = None
        if clip:
            for cr in clip.crop_regions:
                if cr.id == crop_id:
                    target = cr
                    break
        if target and target.group_id:
            ret = QMessageBox.question(
                self, "Remove Crop",
                "Remove this crop region?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        if self._selected_crop_id == crop_id:
            self._selected_crop_id = None
            self.crop_selected.emit("")
        self._timeline.remove_crop_region(self._current_clip.id, crop_id)

    def _on_row_selected(self, crop_id: str):
        # Re-clicking the already-selected crop toggles it off.
        if self._selected_crop_id == crop_id:
            self._apply_crop_selection(None)
        else:
            self._apply_crop_selection(crop_id)

    # ------------------------------------------------------------------
    # + Add crop / Edit-mode toggle
    # ------------------------------------------------------------------

    def _on_edit_toggled(self, checked: bool):
        self._edit_btn.setText(
            "Editing crops in preview" if checked
            else "Edit crops in preview")
        self.crop_edit_mode_changed.emit(checked)

    def _on_add_crop(self):
        clip = self._current_clip
        source = self._current_source
        if (clip is None or clip.is_gap or source is None
                or source.fps <= 0 or not can_host_crop(clip, source.fps)):
            return
        # Default geometry: centered, 50% of source width × source height,
        # clamped to source bounds. Aspect ratio "free".
        sw = max(1, int(source.width))
        sh = max(1, int(source.height))
        cw = max(16, sw // 2)
        ch = max(16, sh // 2)
        cx = (sw - cw) // 2
        cy = (sh - ch) // 2
        # Default first segment: playhead's source frame, clamped to the
        # valid range for this clip's source FPS.
        anchor = clamp_anchor(
            self._playhead_source_frame, clip, source.fps)
        cr = CropRegion(
            x=cx, y=cy, w=cw, h=ch,
            segments=[Segment(anchor_frame=anchor)],
            aspect_ratio="free",
        )
        new_id = self._timeline.add_crop_region(clip.id, cr)
        if new_id:
            self._selected_crop_id = new_id
            self.crop_selected.emit(new_id)
