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

    def __init__(self, clip_count: int = 0, total_frames: int = 0,
                 fps: float = 24.0, has_render_range: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export OpenTimelineIO")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        duration_secs = total_frames / fps if fps > 0 else 0
        mins, secs = divmod(int(duration_secs), 60)
        info = f"{clip_count} clips  |  {total_frames:,} frames  |  {mins}:{secs:02d}"
        info_label = QLabel(info)
        info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(info_label)

        self._gaps_check = QCheckBox("Include gaps between clips")
        self._gaps_check.setToolTip(
            "Checked: gaps appear as empty space in the NLE timeline.\n"
            "Unchecked: clips are placed back-to-back (compact)."
        )
        layout.addWidget(self._gaps_check)

        self._range_check = QCheckBox("Use in/out render range")
        self._range_check.setToolTip(
            "Only export clips within the in/out point range."
        )
        self._range_check.setEnabled(has_render_range)
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
        })
        self._status.setText("OTIO exported!")
