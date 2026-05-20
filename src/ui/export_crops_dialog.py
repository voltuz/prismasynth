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
    QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from core.crop_region import required_source_frames
from core.crop_exporter import IMAGE_SEQUENCE_CODECS, PNG_SEQUENCE_CODEC
from core.timeline import TimelineModel
from core.ui_scale import ui_scale
from ui.export_dialog import AUDIO_FORMAT_PRESETS, IMAGE_FORMATS, VIDEO_PRESETS
from ui.group_filter_widget import GroupFilterWidget


# Codec radio-button grouping for the Video tab — mirrors export_dialog
# so the same labels and ordering are presented to the user across both
# export paths.
_CODEC_GROUPS = [
    ("H.264", [
        ("h264_nvenc", "NVENC (GPU)"),
        ("h264", "Software"),
    ]),
    ("H.265 / HEVC", [
        ("h265_nvenc", "NVENC (GPU)"),
        ("h265", "Software"),
    ]),
    ("ProRes", [
        ("prores_proxy", "422 Proxy"),
        ("prores_lt", "422 LT"),
        ("prores_standard", "422"),
        ("prores_hq", "422 HQ"),
        ("prores_4444", "4444"),
        ("prores_4444xq", "4444 XQ"),
    ]),
    ("Lossless", [
        ("ffv1", "FFV1"),
    ]),
]


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

        # Tabs: Video / Image Sequence. Resolution and FPS are fixed at
        # the crop rectangle's native pixels @ 16fps, so neither tab
        # exposes those controls (unlike the normal export dialog).
        self._tabs = QTabWidget()

        # --- Video tab ----------------------------------------------------
        video_tab = QWidget()
        vl = QVBoxLayout(video_tab)
        vl.setContentsMargins(s.px(6), s.px(6), s.px(6), s.px(6))

        # Codec radio buttons — same layout as the normal Video tab.
        self._codec_group = QButtonGroup(self)
        self._codec_buttons: Dict[str, QRadioButton] = {}
        codec_row = QHBoxLayout()
        codec_row.setSpacing(s.px(12))
        for group_name, codecs in _CODEC_GROUPS:
            box = QGroupBox(group_name)
            box_l = QVBoxLayout(box)
            box_l.setSpacing(2)
            box_l.setContentsMargins(8, 8, 8, 6)
            for key, label in codecs:
                rb = QRadioButton(label)
                rb.setProperty("codec_key", key)
                self._codec_group.addButton(rb)
                self._codec_buttons[key] = rb
                box_l.addWidget(rb)
            codec_row.addWidget(box)
        default_codec = "h264_nvenc" if _has_nvenc() else "h264"
        self._codec_buttons[default_codec].setChecked(True)
        self._codec_group.buttonClicked.connect(self._on_codec_changed)
        vl.addLayout(codec_row)

        # Quality + audio mode for video.
        vform = QFormLayout()
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 51)
        self._quality_spin.setValue(17)
        self._quality_spin.setToolTip(
            "Quality (lower = better, 0 = lossless / max)")
        self._quality_label = QLabel("Quality (CRF/CQ):")
        vform.addRow(self._quality_label, self._quality_spin)

        # Audio mode — embedded / none / both, mirroring the normal
        # dialog's Video tab. 'both' reveals the sidecar format combo.
        self._audio_mode_combo = QComboBox()
        self._audio_mode_combo.addItem("Embedded (default)", "embedded")
        self._audio_mode_combo.addItem("No audio", "none")
        self._audio_mode_combo.addItem(
            "Embedded + sidecar audio file", "both")
        self._audio_mode_combo.currentIndexChanged.connect(
            self._on_audio_mode_changed)
        vform.addRow("Audio mode:", self._audio_mode_combo)

        self._audio_format_label = QLabel("Sidecar format:")
        self._audio_format_combo = QComboBox()
        for key in ("wav", "flac", "mp3", "m4a"):
            self._audio_format_combo.addItem(
                AUDIO_FORMAT_PRESETS[key]["name"], key)
        vform.addRow(self._audio_format_label, self._audio_format_combo)

        vl.addLayout(vform)
        vl.addStretch(1)
        self._tabs.addTab(video_tab, "Video")

        # --- Image Sequence tab ------------------------------------------
        img_tab = QWidget()
        il = QVBoxLayout(img_tab)
        il.setContentsMargins(s.px(6), s.px(6), s.px(6), s.px(6))
        iform = QFormLayout()
        self._img_format_combo = QComboBox()
        for key, fmt in IMAGE_FORMATS.items():
            self._img_format_combo.addItem(fmt["name"], key)
        iform.addRow("Format:", self._img_format_combo)
        il.addLayout(iform)
        il.addWidget(QLabel(
            "Each crop becomes a folder containing 81 frames named "
            "frame_001.<ext> … frame_081.<ext>."))
        il.addStretch(1)
        self._tabs.addTab(img_tab, "Image Sequence")

        outer.addWidget(self._tabs)
        self._on_codec_changed()
        self._on_audio_mode_changed()

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

    def _selected_video_codec(self) -> str:
        """Currently-checked codec radio on the Video tab.
        Falls back to 'h264' if nothing is checked (shouldn't happen)."""
        btn = self._codec_group.checkedButton()
        return btn.property("codec_key") if btn else "h264"

    def _on_codec_changed(self):
        codec_id = self._selected_video_codec()
        preset = VIDEO_PRESETS.get(codec_id, {})
        args = preset.get("args", [])
        uses_quality = any("{quality}" in a for a in args)
        self._quality_spin.setVisible(uses_quality)
        self._quality_label.setVisible(uses_quality)

    def _on_audio_mode_changed(self):
        mode = self._audio_mode_combo.currentData()
        show_sidecar = (mode == "both")
        self._audio_format_label.setVisible(show_sidecar)
        self._audio_format_combo.setVisible(show_sidecar)

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
        # One output file per ACTIVE segment of each exportable crop.
        count = 0
        for _clip, cr in self._timeline.iter_crops():
            if not cr.active:
                continue
            if not crop_matches_filter(cr, flt):
                continue
            count += sum(1 for seg in cr.segments if seg.active)
        if count == 0:
            self._summary_label.setText(
                "No segments match the current filter. Add crops / segments "
                "in the Clip panel.")
        else:
            self._summary_label.setText(
                f"Will export {count} segment"
                f"{'s' if count != 1 else ''} — "
                f"each is 81 frames @ 16 fps "
                f"(native crop pixels, with audio).")

    # ------------------------------------------------------------------

    def _collect_settings(self) -> Optional[dict]:
        # Active tab decides the export shape. The crop_exporter still
        # discriminates on the codec id (image-seq codec keys live in
        # IMAGE_SEQUENCE_CODECS), so passing mode is purely informational
        # for downstream code; the codec id is the source of truth.
        is_image_seq = (self._tabs.currentIndex() == 1)
        if is_image_seq:
            fmt_key = self._img_format_combo.currentData()
            codec = f"{fmt_key}_sequence"
            if codec not in IMAGE_SEQUENCE_CODECS:
                # Fallback: PNG is always supported.
                codec = PNG_SEQUENCE_CODEC
            mode = "image_sequence"
            audio_mode = "none"
            audio_format = "wav"
        else:
            codec = self._selected_video_codec()
            mode = "video"
            audio_mode = self._audio_mode_combo.currentData() or "embedded"
            audio_format = self._audio_format_combo.currentData() or "wav"
        quality = self._quality_spin.value()
        flt = self._group_filter.current_filter()
        is_root_mode = self._rb_root.isChecked()
        common = {
            "codec": codec,
            "mode": mode,
            "quality": quality,
            "audio_mode": audio_mode,
            "audio_format": audio_format,
            "group_filter": flt,
        }
        if is_root_mode:
            root_dir = self._root_edit.text().strip()
            if not root_dir:
                self.set_status("Choose a root folder first.")
                return None
            try:
                os.makedirs(root_dir, exist_ok=True)
            except OSError as e:
                self.set_status(f"Couldn't create root folder: {e}")
                return None
            return {
                **common,
                "output_mode": "root_subfolders",
                "root_dir": root_dir,
                "per_group_paths": {},
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
            **common,
            "output_mode": "per_group_paths",
            "root_dir": "",
            "per_group_paths": per_group_paths,
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
