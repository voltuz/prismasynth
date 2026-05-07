from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QFileDialog, QLineEdit, QFormLayout,
)
from PySide6.QtCore import Signal


class OtioDialog(QDialog):
    """Dialog for OpenTimelineIO (.otio) export settings.

    Native OTIO JSON — imported directly by DaVinci Resolve Studio and any
    OTIO-aware tooling. For Premiere Pro, convert to AAF / Premiere XML
    with an OTIO adapter after export.
    """

    export_requested = Signal(dict)

    def __init__(self, timeline, sources: dict, fps: float = 24.0,
                 has_render_range: bool = False, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._fps = fps
        self.setWindowTitle("Export OpenTimelineIO")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._info_label)

        self._audio_label = QLabel("")
        layout.addWidget(self._audio_label)

        self._gaps_check = QCheckBox("Include gaps between clips")
        self._gaps_check.setToolTip(
            "Checked: gaps appear as empty space in the NLE timeline.\n"
            "Unchecked: clips are placed back-to-back (compact)."
        )
        self._gaps_check.stateChanged.connect(self._refresh_info)
        layout.addWidget(self._gaps_check)

        self._range_check = QCheckBox("Use in/out render range")
        self._range_check.setToolTip(
            "Only export clips within the in/out point range."
        )
        self._range_check.setEnabled(has_render_range)
        self._range_check.stateChanged.connect(self._refresh_info)
        layout.addWidget(self._range_check)

        # People-group filter (hidden when there are no groups).
        from ui.group_filter_widget import GroupFilterWidget
        self._group_filter_widget = GroupFilterWidget(timeline)
        self._group_filter_widget.selection_changed.connect(self._refresh_info)
        layout.addWidget(self._group_filter_widget)

        form = QFormLayout()
        self._output = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        out_layout = QHBoxLayout()
        out_layout.addWidget(self._output, 1)
        out_layout.addWidget(browse_btn)
        form.addRow("Output:", out_layout)
        layout.addLayout(form)

        self._status = QLabel("")
        layout.addWidget(self._status)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export)
        btn_layout.addWidget(self._export_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._refresh_info()

    def _refresh_info(self):
        include_gaps = self._gaps_check.isChecked()
        use_range = self._range_check.isChecked() and self._range_check.isEnabled()
        gf = self._group_filter_widget.current_filter()
        clips, frames = self._timeline.compute_export_extent(
            include_gaps, use_range, group_filter=gf)
        secs_total = int(frames / self._fps) if self._fps > 0 else 0
        h, rem = divmod(secs_total, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        if clips <= 0:
            self._info_label.setText("Nothing to export — adjust the group filter.")
            self._info_label.setStyleSheet("color: #e8a735;")
            self._export_btn.setEnabled(False)
        else:
            self._info_label.setText(
                f"{clips} clips  |  {frames:,} frames  |  {time_str}"
            )
            self._info_label.setStyleSheet("color: #aaa;")
            self._export_btn.setEnabled(True)
        audio = self._timeline.get_export_audio_summary(
            self._sources, use_range, group_filter=gf)
        self._audio_label.setText(f"Audio: {audio}")
        is_silent = (audio == "none")
        self._audio_label.setStyleSheet(
            "color: #e8a735; font-size: 11px;" if is_silent
            else "color: #aaa; font-size: 11px;"
        )

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save OpenTimelineIO", "", "OpenTimelineIO (*.otio)")
        if path:
            self._output.setText(path)

    def _export(self):
        output = self._output.text().strip()
        if not output:
            self._status.setText("Please specify an output path.")
            return
        self.export_requested.emit({
            "output_path": output,
            "include_gaps": self._gaps_check.isChecked(),
            "use_render_range": self._range_check.isChecked(),
            "group_filter": self._group_filter_widget.current_filter(),
        })
        self._status.setText("OTIO exported!")
