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

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from core.clip import Clip
from core.crop_region import (
    ASPECT_PRESETS, CropRegion, can_host_crop, clamp_anchor,
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


class _CropRow(QFrame):
    """One row in the cropping list: eye / label / aspect / group /
    trash. Window position (formerly an 'anchor frame' spinbox) is now
    edited via the timeline strip's drag interaction."""

    active_toggled = Signal(str)
    label_changed = Signal(str, str)
    aspect_changed = Signal(str, str, int, int)       # crop_id, preset, cw, ch
    group_changed = Signal(str, object)               # crop_id, group_id or None
    delete_requested = Signal(str)
    selected = Signal(str)
    # Keyframe controls.
    keyframe_toggle_requested = Signal(str, str)      # crop_id, group ("position"/"size")
    jump_to_source_frame = Signal(str, int)           # crop_id, source_frame
    edit_curves_requested = Signal(str)               # crop_id

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

        # Row 2: keyframe controls (prev/diamond/next per group)
        # + Edit curves button.
        kf_row = QHBoxLayout()
        kf_row.setSpacing(s.px(2))
        kf_row.setContentsMargins(0, 0, 0, 0)
        (self._pos_prev, self._pos_diamond, self._pos_next) = (
            self._build_kf_triplet("position", "Position"))
        kf_row.addWidget(self._pos_prev)
        kf_row.addWidget(self._pos_diamond)
        kf_row.addWidget(self._pos_next)
        kf_row.addSpacing(s.px(6))
        (self._sz_prev, self._sz_diamond, self._sz_next) = (
            self._build_kf_triplet("size", "Size"))
        kf_row.addWidget(self._sz_prev)
        kf_row.addWidget(self._sz_diamond)
        kf_row.addWidget(self._sz_next)
        kf_row.addStretch(1)
        self._edit_curves_btn = QPushButton("Edit curves…")
        self._edit_curves_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._edit_curves_btn.clicked.connect(
            lambda: self.edit_curves_requested.emit(self._crop_id))
        kf_row.addWidget(self._edit_curves_btn)
        layout.addLayout(kf_row, 2, 0, 1, 6)

        # KF visual state. Stays as the "empty" state until ClipInfoPanel
        # pushes a refresh.
        self._refresh_kf_visuals("empty", "empty",
                                False, False, False, False)

        self.update_from(crop, groups)

    # --- KF helpers --------------------------------------------------

    def _build_kf_triplet(self, group: str, label: str):
        """Build (prev_arrow, diamond_btn, next_arrow) for a KF group."""
        s = ui_scale()
        prev_btn = QToolButton()
        prev_btn.setText("◀")
        prev_btn.setFixedSize(s.px(18), s.px(18))
        prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        prev_btn.setToolTip(f"Previous {label} keyframe")
        prev_btn.clicked.connect(
            lambda _=False, g=group: self._on_kf_jump(g, direction=-1))

        diamond_btn = QToolButton()
        diamond_btn.setText("◆")
        diamond_btn.setFixedSize(s.px(36), s.px(18))
        diamond_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        diamond_btn.setToolTip(
            f"Toggle {label} keyframe at playhead "
            "(filled = key here, half = key elsewhere, empty = none)")
        diamond_btn.clicked.connect(
            lambda _=False, g=group:
            self.keyframe_toggle_requested.emit(self._crop_id, g))

        next_btn = QToolButton()
        next_btn.setText("▶")
        next_btn.setFixedSize(s.px(18), s.px(18))
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setToolTip(f"Next {label} keyframe")
        next_btn.clicked.connect(
            lambda _=False, g=group: self._on_kf_jump(g, direction=+1))
        return (prev_btn, diamond_btn, next_btn)

    def _on_kf_jump(self, group: str, direction: int):
        # Cached on the row by ClipInfoPanel.refresh_keyframe_states.
        target = (self._jump_targets.get((group, direction))
                  if hasattr(self, "_jump_targets") else None)
        if target is None:
            return
        self.jump_to_source_frame.emit(self._crop_id, int(target))

    def refresh_kf_state(self, pos_state: str, sz_state: str,
                         pos_prev: Optional[int], pos_next: Optional[int],
                         sz_prev: Optional[int], sz_next: Optional[int]):
        """Push the live keyframe state of this row's crop. ``*_state`` is
        one of ``"empty"`` / ``"half"`` / ``"on"``. ``*_prev`` /
        ``*_next`` are source-frame jump targets or None when no key
        exists on that side."""
        self._jump_targets = {
            ("position", -1): pos_prev,
            ("position", +1): pos_next,
            ("size", -1): sz_prev,
            ("size", +1): sz_next,
        }
        self._refresh_kf_visuals(
            pos_state, sz_state,
            pos_prev is not None, pos_next is not None,
            sz_prev is not None, sz_next is not None,
        )

    @staticmethod
    def _diamond_style(state: str) -> str:
        # Color codes: empty=grey, half=warning yellow, on=orange.
        color = {"empty": "#666666",
                 "half": "#d1a72c",
                 "on": "#e8a735"}.get(state, "#666666")
        return ("QToolButton { background: transparent; color: "
                f"{color}; border: 1px solid #444; border-radius: 3px;"
                " font-weight: bold; }"
                "QToolButton:hover { border-color: #888; }")

    def _refresh_kf_visuals(self, pos_state: str, sz_state: str,
                            pos_prev: bool, pos_next: bool,
                            sz_prev: bool, sz_next: bool):
        self._pos_diamond.setStyleSheet(self._diamond_style(pos_state))
        self._sz_diamond.setStyleSheet(self._diamond_style(sz_state))
        self._pos_prev.setEnabled(pos_prev)
        self._pos_next.setEnabled(pos_next)
        self._sz_prev.setEnabled(sz_prev)
        self._sz_next.setEnabled(sz_next)

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

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources: Dict[str, VideoSource] = {}
        self._current_clip: Optional[Clip] = None
        self._current_source: Optional[VideoSource] = None
        self._crop_rows: Dict[str, _CropRow] = {}
        self._selected_crop_id: Optional[str] = None
        # Playhead source frame is set by MainWindow on every playhead change
        # so "From playhead" can fill the anchor without a round-trip.
        self._playhead_source_frame: int = 0

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

        # Refresh hooks — clips_changed covers crop add/remove/edit; we also
        # refresh on groups_changed so the group combos in each row stay
        # in sync with the registry.
        self._timeline.clips_changed.connect(self._refresh_crops)
        self._timeline.groups_changed.connect(self._refresh_crops)

        self._update_crop_section_enabled()

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
        """MainWindow calls this on every playhead change."""
        self._playhead_source_frame = int(frame)
        self._refresh_keyframe_states()

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
        """External callers (preview overlay) set the selected crop here."""
        if crop_id == "":
            crop_id = None
        if self._selected_crop_id == crop_id:
            return
        self._selected_crop_id = crop_id
        for cid, row in self._crop_rows.items():
            row.set_selected(cid == crop_id)

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
            return

        # Re-resolve in case the live model object differs from the cached
        # reference (snapshot/restore swaps instances).
        live = self._timeline.get_clip_by_id(clip.id)
        if live is None:
            self._update_crop_section_enabled()
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
            row.keyframe_toggle_requested.connect(
                self._on_row_keyframe_toggle)
            row.jump_to_source_frame.connect(
                self._on_row_jump_to_source_frame)
            row.edit_curves_requested.connect(
                self._on_row_edit_curves)
            idx = self._rows_layout.count() - 1
            self._rows_layout.insertWidget(idx, row)
            self._crop_rows[cr.id] = row
            if cr.id == self._selected_crop_id:
                row.set_selected(True)

        self._refresh_keyframe_states()
        self._update_crop_section_enabled()

    # ------------------------------------------------------------------
    # Keyframe state refresh (playhead-driven)
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_group_state(track_a, track_b, frame: int) -> str:
        """Diamond visual state for an axis pair. 'on' iff BOTH axes
        have a key at this frame; 'half' if either has any key; else
        'empty'."""
        if not track_a and not track_b:
            return "empty"
        has_here = (track_a.has_key_at(frame) and track_b.has_key_at(frame))
        if has_here:
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

    def _refresh_keyframe_states(self):
        """Recompute every row's KF visual + nav-enable state. Cheap —
        runs on playhead moves and on clips_changed."""
        clip = self._current_clip
        if clip is None or clip.is_gap:
            return
        live = self._timeline.get_clip_by_id(clip.id)
        if live is None:
            return
        frame = int(self._playhead_source_frame)
        for cr in live.crop_regions:
            row = self._crop_rows.get(cr.id)
            if row is None:
                continue
            pos_state = self._combine_group_state(
                cr.x_track, cr.y_track, frame)
            sz_state = self._combine_group_state(
                cr.w_track, cr.h_track, frame)
            row.refresh_kf_state(
                pos_state, sz_state,
                self._group_prev(cr.x_track, cr.y_track, frame),
                self._group_next(cr.x_track, cr.y_track, frame),
                self._group_prev(cr.w_track, cr.h_track, frame),
                self._group_next(cr.w_track, cr.h_track, frame),
            )

    # ------------------------------------------------------------------
    # Row keyframe callbacks
    # ------------------------------------------------------------------

    def _on_row_keyframe_toggle(self, crop_id: str, group: str):
        if self._current_clip is None:
            return
        self._timeline.toggle_crop_keyframe(
            self._current_clip.id, crop_id, group,
            int(self._playhead_source_frame))

    def _on_row_jump_to_source_frame(self, crop_id: str, source_frame: int):
        if self._current_clip is None:
            return
        self.crop_jump_to_source_frame.emit(
            self._current_clip.id, crop_id, int(source_frame))

    def _on_row_edit_curves(self, crop_id: str):
        if self._current_clip is None:
            return
        self.crop_edit_curves_requested.emit(
            self._current_clip.id, crop_id)

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
        if self._selected_crop_id == crop_id:
            return
        self._selected_crop_id = crop_id
        for cid, row in self._crop_rows.items():
            row.set_selected(cid == crop_id)
        self.crop_selected.emit(crop_id)

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
        # Default anchor: playhead's source frame, clamped to the valid
        # range for this clip's source FPS.
        anchor = clamp_anchor(
            self._playhead_source_frame, clip, source.fps)
        cr = CropRegion(
            x=cx, y=cy, w=cw, h=ch,
            anchor_frame=anchor,
            aspect_ratio="free",
        )
        new_id = self._timeline.add_crop_region(clip.id, cr)
        if new_id:
            self._selected_crop_id = new_id
            self.crop_selected.emit(new_id)
