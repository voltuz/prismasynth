import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QGroupBox, QLineEdit, QTabWidget, QWidget,
)
from PySide6.QtCore import Signal


VIDEO_PRESETS = {
    "h264": {
        "name": "H.264 (MP4)",
        "ext": ".mp4",
        "args": ["-c:v", "libx264", "-preset", "medium", "-crf", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "h265": {
        "name": "H.265/HEVC (MP4)",
        "ext": ".mp4",
        "args": ["-c:v", "libx265", "-preset", "medium", "-crf", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "ffv1": {
        "name": "FFV1 Lossless (MKV)",
        "ext": ".mkv",
        "args": ["-c:v", "ffv1", "-level", "3", "-slicecrc", "1", "-pix_fmt", "yuv444p10le"],
    },
}

IMAGE_FORMATS = {
    "png": {"name": "PNG (Lossless)", "ext": ".png"},
    "jpg": {"name": "JPEG", "ext": ".jpg"},
    "exr": {"name": "OpenEXR (HDR)", "ext": ".exr"},
}


class ExportDialog(QDialog):
    """Export settings dialog for video and image sequence export."""

    export_requested = Signal(dict)  # settings dict

    def __init__(self, default_width: int = 1920, default_height: int = 1080,
                 default_fps: float = 24.0, total_frames: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Tabs for Video vs Image Sequence
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # --- Video tab ---
        video_tab = QWidget()
        vl = QVBoxLayout(video_tab)
        vform = QFormLayout()

        self._codec_combo = QComboBox()
        for key, preset in VIDEO_PRESETS.items():
            self._codec_combo.addItem(preset["name"], key)
        vform.addRow("Codec:", self._codec_combo)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 51)
        self._quality_spin.setValue(18)
        self._quality_spin.setToolTip("CRF quality (lower = better, 0 = lossless). Ignored for FFV1.")
        vform.addRow("Quality (CRF):", self._quality_spin)

        self._vid_width = QSpinBox()
        self._vid_width.setRange(128, 7680)
        self._vid_width.setValue(default_width)
        self._vid_height = QSpinBox()
        self._vid_height.setRange(128, 4320)
        self._vid_height.setValue(default_height)
        res_layout = QHBoxLayout()
        res_layout.addWidget(self._vid_width)
        res_layout.addWidget(QLabel("x"))
        res_layout.addWidget(self._vid_height)
        vform.addRow("Resolution:", res_layout)

        self._vid_fps = QDoubleSpinBox()
        self._vid_fps.setRange(1, 120)
        self._vid_fps.setDecimals(3)
        self._vid_fps.setValue(default_fps)
        vform.addRow("FPS:", self._vid_fps)

        self._vid_output = QLineEdit()
        vid_browse_btn = QPushButton("Browse...")
        vid_browse_btn.clicked.connect(self._browse_video_output)
        out_layout = QHBoxLayout()
        out_layout.addWidget(self._vid_output, 1)
        out_layout.addWidget(vid_browse_btn)
        vform.addRow("Output:", out_layout)

        vl.addLayout(vform)
        self._tabs.addTab(video_tab, "Video")

        # --- Image Sequence tab ---
        img_tab = QWidget()
        il = QVBoxLayout(img_tab)
        iform = QFormLayout()

        self._img_format_combo = QComboBox()
        for key, fmt in IMAGE_FORMATS.items():
            self._img_format_combo.addItem(fmt["name"], key)
        iform.addRow("Format:", self._img_format_combo)

        self._img_width = QSpinBox()
        self._img_width.setRange(128, 7680)
        self._img_width.setValue(default_width)
        self._img_height = QSpinBox()
        self._img_height.setRange(128, 4320)
        self._img_height.setValue(default_height)
        img_res_layout = QHBoxLayout()
        img_res_layout.addWidget(self._img_width)
        img_res_layout.addWidget(QLabel("x"))
        img_res_layout.addWidget(self._img_height)
        iform.addRow("Resolution:", img_res_layout)

        self._img_output_dir = QLineEdit()
        img_browse_btn = QPushButton("Browse...")
        img_browse_btn.clicked.connect(self._browse_image_output)
        img_out_layout = QHBoxLayout()
        img_out_layout.addWidget(self._img_output_dir, 1)
        img_out_layout.addWidget(img_browse_btn)
        iform.addRow("Output Folder:", img_out_layout)

        il.addLayout(iform)
        self._tabs.addTab(img_tab, "Image Sequence")

        # Info
        info_label = QLabel(f"Total frames to export: {total_frames}")
        info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(info_label)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._start_export)
        btn_layout.addWidget(self._export_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _browse_video_output(self):
        codec_key = self._codec_combo.currentData()
        ext = VIDEO_PRESETS[codec_key]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", f"Video (*{ext})"
        )
        if path:
            self._vid_output.setText(path)

    def _browse_image_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._img_output_dir.setText(path)

    def _start_export(self):
        if self._tabs.currentIndex() == 0:
            # Video export
            output = self._vid_output.text().strip()
            if not output:
                self._status_label.setText("Please specify an output path.")
                return
            codec_key = self._codec_combo.currentData()
            preset = VIDEO_PRESETS[codec_key]
            quality = self._quality_spin.value()
            args = [a.replace("{quality}", str(quality)) for a in preset["args"]]
            settings = {
                "mode": "video",
                "output_path": output,
                "codec_key": codec_key,
                "ffmpeg_args": args,
                "ext": preset["ext"],
                "width": self._vid_width.value(),
                "height": self._vid_height.value(),
                "fps": self._vid_fps.value(),
            }
        else:
            # Image sequence export
            output_dir = self._img_output_dir.text().strip()
            if not output_dir:
                self._status_label.setText("Please specify an output folder.")
                return
            fmt_key = self._img_format_combo.currentData()
            settings = {
                "mode": "image_sequence",
                "output_dir": output_dir,
                "format": fmt_key,
                "ext": IMAGE_FORMATS[fmt_key]["ext"],
                "width": self._img_width.value(),
                "height": self._img_height.value(),
            }

        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self.export_requested.emit(settings)

    def set_progress(self, pct: int):
        self._progress.setValue(pct)

    def set_status(self, msg: str):
        self._status_label.setText(msg)

    def export_finished(self):
        self._status_label.setText("Export complete!")
        self._export_btn.setEnabled(True)
        self._progress.setValue(100)
