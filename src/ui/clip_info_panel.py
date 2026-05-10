from typing import Optional, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, QGroupBox, QSizePolicy,
)
from PySide6.QtCore import Qt

from core.clip import Clip
from core.video_source import VideoSource
from core.ui_scale import ui_scale


class ClipInfoPanel(QWidget):
    """Panel showing info about the currently selected clip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Resizable in a horizontal QSplitter; Preferred lets the user drag
        # the splitter handle, with a sensible floor.
        s = ui_scale()
        self.setMinimumWidth(s.px(180))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(s.px(8), s.px(8), s.px(8), s.px(8))
        self._outer_layout = layout
        ui_scale().changed.connect(self._on_ui_scale_changed)

        self._title = QLabel("No clip selected")
        self._title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ddd;")
        layout.addWidget(self._title)

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
                     self._duration_label, self._resolution_label, self._fps_label,
                     self._audio_label]:
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

        layout.addStretch()

    def _on_ui_scale_changed(self):
        s = ui_scale()
        self.setMinimumWidth(s.px(180))
        self._outer_layout.setContentsMargins(
            s.px(8), s.px(8), s.px(8), s.px(8))

    def update_clip(self, clip: Optional[Clip], sources: Dict[str, VideoSource]):
        if clip is None:
            self._title.setText("No clip selected")
            for lbl in [self._source_label, self._in_label, self._out_label,
                         self._duration_label, self._resolution_label, self._fps_label,
                         self._audio_label]:
                lbl.setText("-")
                lbl.setStyleSheet("color: #ccc;")
            return

        if clip.is_gap:
            self._title.setText("Gap")
            self._source_label.setText("-")
            self._in_label.setText("-")
            self._out_label.setText("-")
            self._duration_label.setText(f"{clip.duration_frames} frames")
            self._resolution_label.setText("-")
            self._fps_label.setText("-")
            self._audio_label.setText("-")
            self._audio_label.setStyleSheet("color: #ccc;")
            return

        source = sources.get(clip.source_id)
        source_name = source.file_path.split("\\")[-1].split("/")[-1] if source else "Unknown"
        fps = source.fps if source else 24.0

        self._title.setText(f"Clip {clip.id[:8]}")
        self._source_label.setText(source_name)
        self._in_label.setText(str(clip.source_in))
        self._out_label.setText(str(clip.source_out))

        dur_frames = clip.duration_frames
        dur_secs = dur_frames / fps if fps > 0 else 0
        self._duration_label.setText(f"{dur_frames} frames ({dur_secs:.2f}s)")

        if source:
            self._resolution_label.setText(f"{source.width}x{source.height}")
            self._fps_label.setText(f"{source.fps:.3f}")
            audio = source.format_audio()
            self._audio_label.setText(audio)
            # Highlight "none" so the user notices video-only sources before exporting
            self._audio_label.setStyleSheet(
                "color: #e8a735;" if audio == "none" else "color: #ccc;"
            )
        else:
            self._resolution_label.setText("-")
            self._fps_label.setText("-")
            self._audio_label.setText("-")
            self._audio_label.setStyleSheet("color: #ccc;")
