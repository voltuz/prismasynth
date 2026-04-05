import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QGroupBox, QLineEdit, QTabWidget, QWidget,
    QCheckBox, QSlider, QButtonGroup, QRadioButton, QFrame,
)
from PySide6.QtCore import Signal, Qt


VIDEO_PRESETS = {
    "h264": {
        "name": "H.264 (MP4)",
        "ext": ".mp4",
        "args": ["-c:v", "libx264", "-preset", "medium", "-crf", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "h264_nvenc": {
        "name": "H.264 NVENC (MP4, GPU)",
        "ext": ".mp4",
        "args": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "h265": {
        "name": "H.265/HEVC (MP4)",
        "ext": ".mp4",
        "args": ["-c:v", "libx265", "-preset", "medium", "-crf", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "h265_nvenc": {
        "name": "H.265 NVENC (MP4, GPU)",
        "ext": ".mp4",
        "args": ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "{quality}", "-pix_fmt", "yuv420p"],
    },
    "prores_proxy": {
        "name": "ProRes 422 Proxy (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "0", "-pix_fmt", "yuv422p10le"],
    },
    "prores_lt": {
        "name": "ProRes 422 LT (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "1", "-pix_fmt", "yuv422p10le"],
    },
    "prores_standard": {
        "name": "ProRes 422 (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "2", "-pix_fmt", "yuv422p10le"],
    },
    "prores_hq": {
        "name": "ProRes 422 HQ (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    },
    "prores_4444": {
        "name": "ProRes 4444 (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"],
    },
    "prores_4444xq": {
        "name": "ProRes 4444 XQ (MOV)",
        "ext": ".mov",
        "args": ["-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuva444p10le"],
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
                 default_fps: float = 24.0, total_frames: int = 0,
                 render_frames: int = None, clip_count: int = 0,
                 source_width: int = 0, source_height: int = 0, parent=None):
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

        # Codec selection — grouped radio buttons
        CODEC_GROUPS = [
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

        self._codec_group = QButtonGroup(self)
        self._codec_buttons = {}  # key -> QRadioButton
        codec_layout = QHBoxLayout()
        codec_layout.setSpacing(12)

        for group_name, codecs in CODEC_GROUPS:
            group_box = QGroupBox(group_name)
            group_vl = QVBoxLayout(group_box)
            group_vl.setSpacing(2)
            group_vl.setContentsMargins(8, 8, 8, 6)
            for key, label in codecs:
                rb = QRadioButton(label)
                rb.setProperty("codec_key", key)
                self._codec_group.addButton(rb)
                self._codec_buttons[key] = rb
                group_vl.addWidget(rb)
            codec_layout.addWidget(group_box)

        # Default to H.264 NVENC
        self._codec_buttons["h264_nvenc"].setChecked(True)
        self._codec_group.buttonClicked.connect(self._on_codec_changed)

        vl.addLayout(codec_layout)

        # Settings below codec selection
        vform = QFormLayout()

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 51)
        self._quality_spin.setValue(18)
        self._quality_spin.setToolTip("Quality (lower = better, 0 = lossless)")
        self._quality_label = QLabel("Quality:")
        vform.addRow(self._quality_label, self._quality_spin)

        # Denoise controls
        self._denoise_check = QCheckBox("Denoise (FastDVDnet)")
        self._denoise_check.setToolTip("AI temporal denoiser — uses 5-frame windows for high-quality noise removal")
        self._denoise_sigma = QSpinBox()
        self._denoise_sigma.setRange(5, 55)
        self._denoise_sigma.setValue(25)
        self._denoise_sigma.setEnabled(False)
        self._denoise_sigma.setToolTip("5-15: light (clean sources)  |  15-30: medium (film grain)  |  30-55: heavy (noisy/low light)")
        self._denoise_check.toggled.connect(self._denoise_sigma.setEnabled)
        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(self._denoise_check)
        denoise_layout.addWidget(QLabel("Strength:"))
        denoise_layout.addWidget(self._denoise_sigma)
        denoise_layout.addStretch()
        vform.addRow("", denoise_layout)

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

        # Info stats
        fps = default_fps if default_fps > 0 else 24.0
        export_frames = render_frames if render_frames is not None else total_frames
        duration_secs = export_frames / fps
        mins, secs = divmod(int(duration_secs), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            duration_str = f"{hours:d}:{mins:02d}:{secs:02d}"
        else:
            duration_str = f"{mins:d}:{secs:02d}"

        if render_frames is not None and render_frames != total_frames:
            frames_str = f"{render_frames:,} frames (in/out range of {total_frames:,})"
        else:
            frames_str = f"{total_frames:,} frames"

        res_str = f"{source_width}x{source_height}" if source_width and source_height else ""
        parts = [f"Clips: {clip_count}", frames_str, f"Duration: {duration_str}"]
        if res_str:
            parts.append(f"Source: {res_str}")
        info_label = QLabel("  |  ".join(parts))
        info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(info_label)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #aaa;")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_label)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # For ETA calculation
        self._export_start_time = 0.0
        self._total_export_frames = render_frames if render_frames is not None else total_frames
        self._export_fps = default_fps if default_fps > 0 else 24.0

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

    def _selected_codec_key(self):
        btn = self._codec_group.checkedButton()
        return btn.property("codec_key") if btn else "h264_nvenc"

    def _on_codec_changed(self):
        key = self._selected_codec_key()
        has_quality = "{quality}" in " ".join(VIDEO_PRESETS[key]["args"])
        self._quality_spin.setVisible(has_quality)
        self._quality_label.setVisible(has_quality)

    def _browse_video_output(self):
        codec_key = self._selected_codec_key()
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
            codec_key = self._selected_codec_key()
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
                "denoise": self._denoise_check.isChecked(),
                "denoise_sigma": self._denoise_sigma.value(),
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
        self._progress_label.setVisible(True)
        import time
        self._export_start_time = time.monotonic()
        self.export_requested.emit(settings)

    def set_progress(self, pct: int):
        self._progress.setValue(pct)
        if pct <= 0 or self._export_start_time <= 0:
            return
        import time
        elapsed = time.monotonic() - self._export_start_time
        total_frames = self._total_export_frames
        fps = self._export_fps
        done_frames = int(total_frames * pct / 100)
        done_secs = done_frames / fps
        total_secs = total_frames / fps

        # ETA
        if pct > 0:
            eta_secs = max(0, elapsed * (100 - pct) / pct)
            eta_m, eta_s = divmod(int(eta_secs), 60)
            eta_str = f"{eta_m:d}:{eta_s:02d}"
        else:
            eta_str = "--:--"

        # Formatted times
        def fmt_time(s):
            m, sec = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"

        # Export speed in fps
        export_fps = done_frames / elapsed if elapsed > 0 else 0

        self._progress_label.setText(
            f"{done_frames:,} / {total_frames:,} frames  |  "
            f"{export_fps:.1f} fps  |  "
            f"Elapsed: {fmt_time(elapsed)}  |  ETA: {eta_str}"
        )

    def set_status(self, msg: str):
        self._status_label.setText(msg)

    def export_finished(self):
        self._status_label.setText("Export complete!")
        self._export_btn.setEnabled(True)
        self._progress.setValue(100)
