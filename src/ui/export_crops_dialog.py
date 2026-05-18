"""Standalone "Export Crops" dialog.

Separate from the unified ``ExportDialog`` because the per-crop output
shape (81 frames @ 16fps, native crop pixels, per-region file, optional
per-group folder routing) is meaningfully different from any timeline
export tab. Reuses ``GroupFilterWidget`` and ``VIDEO_PRESETS`` from the
existing export plumbing so behaviour stays consistent.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from core.crop_region import required_source_frames
from core.crop_exporter import PNG_SEQUENCE_CODEC
from core.timeline import TimelineModel
from core.ui_scale import ui_scale
from ui.export_dialog import VIDEO_PRESETS
from ui.group_filter_widget import GroupFilterWidget


_UNTAGGED_KEY = "__untagged__"


def _has_nvenc() -> bool:
    """Best-effort detect: ffmpeg lists the codec in -encoders. Fall back
    to False on any failure — we'll still default to a working CPU codec."""
    try:
        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, timeout=4,
        )
        return b"h264_nvenc" in (r.stdout or b"")
    except Exception:
        return False


class _PerGroupRow(QWidget):
    """One row in 'Custom path per group' mode: group label + folder picker."""

    path_changed = Signal(str, str)  # group_key, new_path

    def __init__(self, group_key: str, label: str, color: Optional[str],
                 italic: bool, default_dir: str, parent=None):
        super().__init__(parent)
        self._group_key = group_key
        self._default_dir = default_dir
        s = ui_scale()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(s.px(6))
        if color is not None:
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555;"
                f" border-radius: 2px;")
            layout.addWidget(swatch)
        else:
            spacer = QLabel()
            spacer.setFixedSize(12, 12)
            layout.addWidget(spacer)
        name = QLabel(label)
        name.setMinimumWidth(s.px(120))
        if italic:
            f = name.font()
            f.setItalic(True)
            name.setFont(f)
            name.setStyleSheet("color: #aaa;")
        layout.addWidget(name)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("(no folder set — skip)")
        self._path_edit.textChanged.connect(
            lambda t: self.path_changed.emit(self._group_key, t))
        layout.addWidget(self._path_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        layout.addWidget(browse)

    def set_path(self, path: str):
        self._path_edit.setText(path)

    def current_path(self) -> str:
        return self._path_edit.text().strip()

    def _on_browse(self):
        start = self._path_edit.text().strip() or self._default_dir
        path = QFileDialog.getExistingDirectory(
            self, "Folder for this group's crops", start)
        if path:
            self._path_edit.setText(path)


class ExportCropsDialog(QDialog):
    """Modal dialog driving ``core.crop_exporter.CropExporter``."""

    export_requested = Signal(dict)
    cancel_requested = Signal()

    def __init__(self, timeline: TimelineModel,
                 sources: Dict[str, "VideoSource"],
                 default_dir: str = "",
                 parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._default_dir = default_dir or ""
        self._is_running = False
        self.setWindowTitle("Export Crops")
        self.setMinimumWidth(ui_scale().px(620))
        self.setModal(True)

        s = ui_scale()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(s.px(12), s.px(12), s.px(12), s.px(12))
        outer.setSpacing(s.px(10))

        # Summary line — total crops and a per-clip-fps note.
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("color: #ddd;")
        outer.addWidget(self._summary_label)

        # Group filter (the existing reusable widget).
        self._group_filter = GroupFilterWidget(self._timeline)
        self._group_filter.selection_changed.connect(self._refresh_summary)
        outer.addWidget(self._group_filter)

        # Output mode radios + per-mode controls.
        out_box = QGroupBox("Output")
        out_layout = QVBoxLayout(out_box)
        self._mode_btns = QButtonGroup(self)
        self._rb_root = QRadioButton(
            "One root folder, subfolders named after each group")
        self._rb_per_group = QRadioButton(
            "Custom output path per group")
        self._rb_root.setChecked(True)
        self._mode_btns.addButton(self._rb_root)
        self._mode_btns.addButton(self._rb_per_group)
        self._rb_root.toggled.connect(self._on_mode_changed)
        out_layout.addWidget(self._rb_root)

        root_row = QHBoxLayout()
        root_row.setContentsMargins(s.px(20), 0, 0, 0)
        self._root_edit = QLineEdit(self._default_dir)
        self._root_edit.setPlaceholderText("Root folder for the crops")
        root_row.addWidget(self._root_edit, 1)
        self._root_browse = QPushButton("Browse…")
        self._root_browse.clicked.connect(self._on_browse_root)
        root_row.addWidget(self._root_browse)
        out_layout.addLayout(root_row)

        out_layout.addWidget(self._rb_per_group)
        per_group_host = QWidget()
        per_group_layout = QVBoxLayout(per_group_host)
        per_group_layout.setContentsMargins(s.px(20), 0, 0, 0)
        per_group_layout.setSpacing(s.px(4))
        # Scrollable list of per-group folder pickers.
        self._per_group_rows: Dict[str, _PerGroupRow] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(s.px(160))
        self._per_group_host = QWidget()
        self._per_group_inner = QVBoxLayout(self._per_group_host)
        self._per_group_inner.setContentsMargins(0, 0, 0, 0)
        self._per_group_inner.setSpacing(s.px(2))
        self._per_group_inner.addStretch(1)
        scroll.setWidget(self._per_group_host)
        per_group_layout.addWidget(scroll)
        out_layout.addWidget(per_group_host)
        outer.addWidget(out_box)
        self._timeline.groups_changed.connect(self._rebuild_per_group_rows)
        self._rebuild_per_group_rows()

        # Codec / quality.
        codec_box = QGroupBox("Encoding")
        codec_form = QFormLayout(codec_box)
        self._codec_combo = QComboBox()
        for codec_id, preset in VIDEO_PRESETS.items():
            self._codec_combo.addItem(preset["name"], codec_id)
        self._codec_combo.addItem("PNG image sequence", PNG_SEQUENCE_CODEC)
        # Default = H.264 NVENC if available, else CPU H.264.
        default_codec = "h264_nvenc" if _has_nvenc() else "h264"
        idx = self._codec_combo.findData(default_codec)
        if idx >= 0:
            self._codec_combo.setCurrentIndex(idx)
        self._codec_combo.currentIndexChanged.connect(self._on_codec_changed)
        codec_form.addRow("Codec:", self._codec_combo)
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 51)
        self._quality_spin.setValue(17)
        codec_form.addRow("Quality (CRF/CQ):", self._quality_spin)
        outer.addWidget(codec_box)
        self._on_codec_changed()

        # Status / progress.
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        outer.addWidget(self._progress)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #ccc;")
        outer.addWidget(self._status_label)

        # Action buttons (state-driven, mirrors ExportDialog).
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._run_btn = QPushButton("Export")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._run_btn)
        outer.addLayout(btn_row)

        self._on_mode_changed()
        self._refresh_summary()

    # ------------------------------------------------------------------

    def set_progress(self, pct: int):
        self._progress.setValue(max(0, min(100, int(pct))))

    def set_status(self, text: str):
        self._status_label.setText(text)

    def export_finished(self):
        self._is_running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setText("Close")

    def export_cancelled(self):
        self._is_running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setText("Close")

    def export_errored(self, message: str):
        self._is_running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setText("Close")
        self._status_label.setText(f"Error: {message}")

    # ------------------------------------------------------------------

    def _on_mode_changed(self):
        is_root = self._rb_root.isChecked()
        self._root_edit.setEnabled(is_root)
        self._root_browse.setEnabled(is_root)
        for row in self._per_group_rows.values():
            row.setEnabled(not is_root)

    def _on_codec_changed(self):
        codec_id = self._codec_combo.currentData()
        if codec_id == PNG_SEQUENCE_CODEC:
            self._quality_spin.setEnabled(False)
            return
        preset = VIDEO_PRESETS.get(codec_id, {})
        args = preset.get("args", [])
        uses_quality = any("{quality}" in a for a in args)
        self._quality_spin.setEnabled(uses_quality)

    def _on_browse_root(self):
        start = self._root_edit.text().strip() or self._default_dir
        path = QFileDialog.getExistingDirectory(
            self, "Root folder for crops", start)
        if path:
            self._root_edit.setText(path)

    def _rebuild_per_group_rows(self):
        # Drop existing rows
        for row in list(self._per_group_rows.values()):
            self._per_group_inner.removeWidget(row)
            row.deleteLater()
        self._per_group_rows.clear()

        # "(Untagged)" first, then groups in registry order.
        rows = [(_UNTAGGED_KEY, "(Untagged)", None, True)]
        for gid, g in self._timeline.groups.items():
            rows.append((gid, g.name, g.color, False))
        for key, label, color, italic in rows:
            row = _PerGroupRow(key, label, color, italic,
                               self._default_dir, parent=self._per_group_host)
            idx = self._per_group_inner.count() - 1
            self._per_group_inner.insertWidget(idx, row)
            self._per_group_rows[key] = row
        # Sync enabled state.
        self._on_mode_changed()

    def _refresh_summary(self):
        flt = self._group_filter.current_filter()
        from core.crop_region import crop_matches_filter
        count = 0
        for _clip, cr in self._timeline.iter_crops():
            if not cr.active:
                continue
            if crop_matches_filter(cr, flt):
                count += 1
        if count == 0:
            self._summary_label.setText(
                "No crops match the current filter. Add crops in the Clip panel.")
        else:
            self._summary_label.setText(
                f"Will export {count} crop"
                f"{'s' if count != 1 else ''} — "
                f"each is 81 frames @ 16 fps "
                f"(native crop pixels, with audio).")

    # ------------------------------------------------------------------

    def _collect_settings(self) -> Optional[dict]:
        codec = self._codec_combo.currentData()
        quality = self._quality_spin.value()
        flt = self._group_filter.current_filter()
        is_root_mode = self._rb_root.isChecked()
        if is_root_mode:
            root_dir = self._root_edit.text().strip()
            if not root_dir:
                self.set_status(
                    "Choose a root folder first.")
                return None
            try:
                os.makedirs(root_dir, exist_ok=True)
            except OSError as e:
                self.set_status(f"Couldn't create root folder: {e}")
                return None
            return {
                "codec": codec,
                "quality": quality,
                "output_mode": "root_subfolders",
                "root_dir": root_dir,
                "per_group_paths": {},
                "group_filter": flt,
            }
        per_group_paths: Dict[Optional[str], str] = {}
        for key, row in self._per_group_rows.items():
            path = row.current_path()
            if not path:
                continue
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                self.set_status(
                    f"Couldn't create folder for {key}: {e}")
                return None
            per_group_paths[None if key == _UNTAGGED_KEY else key] = path
        if not per_group_paths:
            self.set_status(
                "Set at least one per-group folder before exporting.")
            return None
        return {
            "codec": codec,
            "quality": quality,
            "output_mode": "per_group_paths",
            "root_dir": "",
            "per_group_paths": per_group_paths,
            "group_filter": flt,
        }

    def _on_run(self):
        settings = self._collect_settings()
        if settings is None:
            return
        self._is_running = True
        self._run_btn.setEnabled(False)
        self._cancel_btn.setText("Cancel Export")
        self._progress.setValue(0)
        self.set_status("Starting…")
        self.export_requested.emit(settings)

    def _on_cancel(self):
        if self._is_running:
            self.cancel_requested.emit()
            return
        self.reject()
