import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QGroupBox, QLineEdit, QTabWidget, QWidget,
    QSlider, QButtonGroup, QRadioButton, QFrame,
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
        "args": ["-c:v", "ffv1", "-level", "3", "-slices", "4", "-slicecrc", "1", "-pix_fmt", "yuv444p10le"],
    },
}

IMAGE_FORMATS = {
    "png": {"name": "PNG (Lossless)", "ext": ".png"},
    "jpg": {"name": "JPEG", "ext": ".jpg"},
    "exr": {"name": "OpenEXR (HDR)", "ext": ".exr"},
}

# Audio formats for standalone audio export. The exporter has its own
# preset map with codec/encoder args; this is just for UI display + ext.
AUDIO_FORMAT_PRESETS = {
    "wav":  {"name": "WAV (PCM)",     "ext": ".wav"},
    "flac": {"name": "FLAC",          "ext": ".flac"},
    "mp3":  {"name": "MP3 (192k)",    "ext": ".mp3"},
    "m4a":  {"name": "M4A (AAC 320k)","ext": ".m4a"},
}


class ExportDialog(QDialog):
    """Export settings dialog for video and image sequence export."""

    export_requested = Signal(dict)  # settings dict
    cancel_requested = Signal()      # user hit Cancel during an active export

    def __init__(self, default_width: int = 1920, default_height: int = 1080,
                 default_fps: float = 24.0, total_frames: int = 0,
                 render_frames: int = None, clip_count: int = 0,
                 source_width: int = 0, source_height: int = 0,
                 timeline=None, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._has_render_range = (
            render_frames is not None and render_frames != total_frames)
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
        self._vid_browse_btn = QPushButton("Browse...")
        self._vid_browse_btn.clicked.connect(self._browse_video_output)
        out_layout = QHBoxLayout()
        out_layout.addWidget(self._vid_output, 1)
        out_layout.addWidget(self._vid_browse_btn)
        vform.addRow("Output:", out_layout)

        vl.addLayout(vform)

        # --- Audio group ---
        audio_group = QGroupBox("Audio")
        ag_form = QFormLayout(audio_group)

        # Audio mode — three options on the Video tab. Standalone audio-only
        # export lives on its own tab + Timeline menu entry.
        self._audio_mode_combo = QComboBox()
        self._audio_mode_combo.addItem("Save audio in video", "embedded")
        self._audio_mode_combo.addItem("No audio", "none")
        self._audio_mode_combo.addItem(
            "Save audio in video + standalone", "both")
        self._audio_mode_combo.setCurrentIndex(0)
        self._audio_mode_combo.currentIndexChanged.connect(
            self._on_audio_mode_changed)
        ag_form.addRow("Mode:", self._audio_mode_combo)

        # Format — visible only when mode == "both" (sidecar file).
        self._audio_format_label = QLabel("Format:")
        self._audio_format_combo = QComboBox()
        for key in ("wav", "flac", "mp3", "m4a"):
            self._audio_format_combo.addItem(
                AUDIO_FORMAT_PRESETS[key]["name"], key)
        self._audio_format_combo.currentIndexChanged.connect(
            self._on_audio_format_changed)
        ag_form.addRow(self._audio_format_label, self._audio_format_combo)

        # Location — "Next to video file" (auto-derived path, no field shown)
        # or "Custom path" (reveals the line edit + Browse). Visible only
        # when mode == "both".
        self._audio_location_label = QLabel("Location:")
        self._audio_location_combo = QComboBox()
        self._audio_location_combo.addItem("Next to video file", "auto")
        self._audio_location_combo.addItem("Custom path", "custom")
        self._audio_location_combo.setCurrentIndex(0)
        self._audio_location_combo.currentIndexChanged.connect(
            self._on_audio_location_changed)
        ag_form.addRow(self._audio_location_label, self._audio_location_combo)

        self._audio_output_label = QLabel("Audio output:")
        self._audio_output = QLineEdit()
        self._audio_browse_btn = QPushButton("Browse...")
        self._audio_browse_btn.clicked.connect(self._browse_audio_output)
        audio_out_layout = QHBoxLayout()
        audio_out_layout.addWidget(self._audio_output, 1)
        audio_out_layout.addWidget(self._audio_browse_btn)
        ag_form.addRow(self._audio_output_label, audio_out_layout)

        vl.addWidget(audio_group)

        # Track the last value we auto-filled into the audio output line so
        # we can tell if the user has manually edited it (don't clobber).
        self._audio_path_auto_value = ""
        # When the user retypes the video output, refresh the custom audio
        # path field if it's empty or still showing the previous derived
        # value. The auto-location case derives at export time so this is
        # purely a UX nicety for the custom field.
        self._vid_output.textChanged.connect(
            lambda _t: self._maybe_autofill_audio_path())

        # Apply initial visibility state (mode = embedded → everything hidden).
        self._on_audio_mode_changed()

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

        # --- Audio tab (audio-only export, no video) ---
        audio_tab = QWidget()
        al = QVBoxLayout(audio_tab)
        aform = QFormLayout()

        self._aud_format_combo = QComboBox()
        for key in ("wav", "flac", "mp3", "m4a"):
            self._aud_format_combo.addItem(
                AUDIO_FORMAT_PRESETS[key]["name"], key)
        self._aud_format_combo.currentIndexChanged.connect(
            self._on_audio_tab_format_changed)
        aform.addRow("Format:", self._aud_format_combo)

        self._aud_output = QLineEdit()
        aud_browse_btn = QPushButton("Browse...")
        aud_browse_btn.clicked.connect(self._browse_audio_tab_output)
        aud_out_layout = QHBoxLayout()
        aud_out_layout.addWidget(self._aud_output, 1)
        aud_out_layout.addWidget(aud_browse_btn)
        aform.addRow("Output:", aud_out_layout)

        al.addLayout(aform)
        self._tabs.addTab(audio_tab, "Audio only")

        # People-group filter — shared across all tabs. Hidden when the
        # project has zero groups.
        self._group_filter_widget = None
        if timeline is not None:
            from ui.group_filter_widget import GroupFilterWidget
            self._group_filter_widget = GroupFilterWidget(timeline)
            self._group_filter_widget.selection_changed.connect(
                self._refresh_info)
            layout.addWidget(self._group_filter_widget)

        # Cached defaults so _refresh_info can recompute under filter changes.
        self._default_clip_count = clip_count
        self._default_total_frames = total_frames
        self._default_render_frames = render_frames
        self._source_width = source_width
        self._source_height = source_height

        # Info stats
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._info_label)

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
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        # Dialog state: "idle" (before run), "running", "done" (finished or
        # cancelled). The cancel button's label and action depend on this.
        self._state = "idle"

        # First-time render of the info bar with whatever filter is active
        # at construction (default: no filter — same numbers as before).
        self._refresh_info()

    def _refresh_info(self):
        """Recompute clip / frame counts under the current group filter and
        render them into the info label. Disables the Export button + shows
        'Nothing to export' when the filter would produce zero clips."""
        gf = (self._group_filter_widget.current_filter()
              if self._group_filter_widget is not None else None)
        if gf is None or self._timeline is None:
            # No filter active — fall back to the precomputed defaults.
            clip_count = self._default_clip_count
            export_frames = (self._default_render_frames
                             if self._default_render_frames is not None
                             else self._default_total_frames)
        else:
            use_range = self._has_render_range
            clip_count, export_frames = self._timeline.compute_export_extent(
                include_gaps=False,
                use_render_range=use_range,
                group_filter=gf)
        # Update the running totals so progress / ETA stay coherent.
        self._total_export_frames = export_frames
        # Format duration / frames / size strings as before.
        fps = self._export_fps if self._export_fps > 0 else 24.0
        duration_secs = export_frames / fps if fps else 0
        mins, secs = divmod(int(duration_secs), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            duration_str = f"{hours:d}:{mins:02d}:{secs:02d}"
        else:
            duration_str = f"{mins:d}:{secs:02d}"
        if (gf is None and self._default_render_frames is not None
                and self._default_render_frames != self._default_total_frames):
            frames_str = (f"{export_frames:,} frames "
                          f"(in/out range of {self._default_total_frames:,})")
        else:
            frames_str = f"{export_frames:,} frames"
        parts = [f"Clips: {clip_count}", frames_str,
                 f"Duration: {duration_str}"]
        if self._source_width and self._source_height:
            parts.append(f"Source: {self._source_width}x{self._source_height}")
        if clip_count <= 0:
            self._info_label.setText("Nothing to export — adjust the group filter.")
            self._info_label.setStyleSheet("color: #e8a735;")
            self._export_btn.setEnabled(False)
        else:
            self._info_label.setText("  |  ".join(parts))
            self._info_label.setStyleSheet("color: #aaa;")
            # Don't re-enable mid-run — only when idle.
            if self._state == "idle":
                self._export_btn.setEnabled(True)

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

    # --- Audio controls ---

    def _on_audio_mode_changed(self):
        mode = self._audio_mode_combo.currentData()
        sidecar = (mode == "both")
        # Format + Location rows are only relevant for the sidecar.
        self._audio_format_label.setVisible(sidecar)
        self._audio_format_combo.setVisible(sidecar)
        self._audio_location_label.setVisible(sidecar)
        self._audio_location_combo.setVisible(sidecar)
        # The custom-path row also depends on which location is chosen.
        self._update_audio_path_row_visibility()
        if sidecar:
            self._maybe_autofill_audio_path()

    def _on_audio_location_changed(self):
        self._update_audio_path_row_visibility()
        # When the user switches to Custom, pre-fill the field from the
        # video output so they don't start from a blank line.
        if self._audio_location_combo.currentData() == "custom":
            self._maybe_autofill_audio_path()

    def _update_audio_path_row_visibility(self):
        mode = self._audio_mode_combo.currentData()
        location = self._audio_location_combo.currentData()
        custom = (mode == "both" and location == "custom")
        self._audio_output_label.setVisible(custom)
        self._audio_output.setVisible(custom)
        self._audio_browse_btn.setVisible(custom)

    def _on_audio_format_changed(self):
        self._maybe_autofill_audio_path()

    def _browse_audio_output(self):
        fmt_key = self._audio_format_combo.currentData() or "wav"
        ext = AUDIO_FORMAT_PRESETS[fmt_key]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", f"Audio (*{ext})"
        )
        if path:
            if not path.lower().endswith(ext):
                path += ext
            self._audio_output.setText(path)
            # User-chosen path: clear the auto-fill marker so the next
            # autofill no-ops (don't clobber).
            self._audio_path_auto_value = path

    def _derive_audio_path(self) -> str:
        """Audio output derived from video output basename + audio format ext.
        Returns '' if there's no video output to derive from."""
        vid_path = self._vid_output.text().strip()
        if not vid_path:
            return ""
        fmt_key = self._audio_format_combo.currentData() or "wav"
        ext = AUDIO_FORMAT_PRESETS[fmt_key]["ext"]
        base, _src_ext = os.path.splitext(vid_path)
        return base + ext

    def _maybe_autofill_audio_path(self):
        """Auto-fill the custom audio output line from the video output path.
        Skipped when the user has edited the line manually (current text !=
        last auto-filled value)."""
        cur = self._audio_output.text().strip()
        if cur and cur != self._audio_path_auto_value:
            return
        derived = self._derive_audio_path()
        if not derived:
            return
        self._audio_output.setText(derived)
        self._audio_path_auto_value = derived

    def _browse_image_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._img_output_dir.setText(path)

    def _browse_audio_tab_output(self):
        fmt_key = self._aud_format_combo.currentData() or "wav"
        ext = AUDIO_FORMAT_PRESETS[fmt_key]["ext"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", f"Audio (*{ext})"
        )
        if path:
            if not path.lower().endswith(ext):
                path += ext
            self._aud_output.setText(path)

    def _on_audio_tab_format_changed(self):
        # Swap the extension of the audio-tab output line if it currently
        # ends in any of the known audio extensions. Leaves user-entered
        # paths alone if they don't match a known ext.
        cur = self._aud_output.text().strip()
        if not cur:
            return
        known_exts = tuple(p["ext"] for p in AUDIO_FORMAT_PRESETS.values())
        if not cur.lower().endswith(known_exts):
            return
        new_ext = AUDIO_FORMAT_PRESETS[
            self._aud_format_combo.currentData() or "wav"]["ext"]
        base, _ = os.path.splitext(cur)
        self._aud_output.setText(base + new_ext)

    def _start_export(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            # Video tab — three audio modes (standalone has its own tab)
            audio_mode = self._audio_mode_combo.currentData() or "embedded"
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
                "audio_mode": audio_mode,
            }
            if audio_mode == "both":
                audio_format = self._audio_format_combo.currentData() or "wav"
                ext = AUDIO_FORMAT_PRESETS[audio_format]["ext"]
                location = self._audio_location_combo.currentData() or "auto"
                if location == "auto":
                    # Sidecar alongside the video, same basename, audio ext.
                    base, _ = os.path.splitext(output)
                    audio_output = base + ext
                else:  # custom
                    audio_output = self._audio_output.text().strip()
                    if not audio_output:
                        self._status_label.setText(
                            "Please specify the custom audio output path.")
                        return
                    if not audio_output.lower().endswith(ext):
                        audio_output = os.path.splitext(audio_output)[0] + ext
                settings["audio_format"] = audio_format
                settings["audio_output_path"] = audio_output
        elif idx == 1:
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
        else:
            # Audio-only export
            audio_format = self._aud_format_combo.currentData() or "wav"
            ext = AUDIO_FORMAT_PRESETS[audio_format]["ext"]
            audio_output = self._aud_output.text().strip()
            if not audio_output:
                self._status_label.setText(
                    "Please specify an audio output path.")
                return
            if not audio_output.lower().endswith(ext):
                audio_output = os.path.splitext(audio_output)[0] + ext
            settings = {
                # Reuses the video dispatch path with audio_mode=standalone
                # so the Exporter routes to _export_audio_only.
                "mode": "video",
                "output_path": "",
                "codec_key": "",
                "ffmpeg_args": [],
                "ext": ext,
                "width": 0,
                "height": 0,
                "fps": 24.0,  # unused for audio-only path
                "audio_mode": "standalone",
                "audio_format": audio_format,
                "audio_output_path": audio_output,
            }

        # Attach the active group filter (None when no checkbox is ticked).
        if self._group_filter_widget is not None:
            settings["group_filter"] = self._group_filter_widget.current_filter()
        else:
            settings["group_filter"] = None

        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress_label.setVisible(True)
        import time
        self._export_start_time = time.monotonic()
        self._set_state("running")
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
        self._set_state("done")

    def export_cancelled(self):
        self._status_label.setText("Export cancelled.")
        self._export_btn.setEnabled(True)
        self._set_state("done")

    def export_failed(self, msg: str):
        # Stay in idle so the user can retry without reopening the dialog.
        self._status_label.setText(f"Error: {msg}")
        self._export_btn.setEnabled(True)
        self._set_state("idle")

    def _on_cancel_clicked(self):
        if self._state == "running":
            # Cancel the active export, keep the dialog open so the user
            # sees the "Cancelling…" / "Export cancelled." status update.
            self._status_label.setText("Cancelling...")
            self._cancel_btn.setEnabled(False)
            self.cancel_requested.emit()
        else:
            # Idle or done — just close.
            self.reject()

    def _set_state(self, state: str):
        self._state = state
        if state == "idle":
            self._cancel_btn.setText("Cancel")
        elif state == "running":
            self._cancel_btn.setText("Cancel Export")
        elif state == "done":
            self._cancel_btn.setText("Close")
        self._cancel_btn.setEnabled(True)
