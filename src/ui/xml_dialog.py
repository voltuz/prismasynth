from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QFileDialog, QLineEdit, QFormLayout,
)
from PySide6.QtCore import Signal


class XmlDialog(QDialog):
    """Dialog for FCPXML export settings (Final Cut Pro XML 1.9).

    Single output format — EDL was removed from PrismaSynth because
    Resolve's NDF timecode interpretation drifts multi-clip EDL imports
    by ±1 frame with no reliable server-side fix. FCPXML uses rational
    time fractions and, with the NTSC nudge in ``xml_exporter``, lands
    every clip frame-exact.
    """

    export_requested = Signal(dict)  # settings dict

    def __init__(self, timeline, fps: float = 24.0,
                 has_render_range: bool = False, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._fps = fps
        self.setWindowTitle("Export XML (FCPXML)")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._info_label)

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
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export)
        btn_layout.addWidget(export_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._refresh_info()

    def _refresh_info(self):
        include_gaps = self._gaps_check.isChecked()
        use_range = self._range_check.isChecked() and self._range_check.isEnabled()
        clips, frames = self._timeline.compute_export_extent(include_gaps, use_range)
        secs_total = int(frames / self._fps) if self._fps > 0 else 0
        h, rem = divmod(secs_total, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        self._info_label.setText(
            f"{clips} clips  |  {frames:,} frames  |  {time_str}"
        )

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FCPXML", "", "Final Cut Pro XML (*.fcpxml)")
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
        })
        self._status.setText("FCPXML exported!")
