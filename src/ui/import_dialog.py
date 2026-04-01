import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog,
)
from PySide6.QtCore import Signal

from core.video_source import VideoSource
from core.clip import Clip
from utils.ffprobe import probe_video

logger = logging.getLogger(__name__)

VIDEO_FILTERS = "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v *.ts *.mxf);;All Files (*)"


class ImportDialog(QDialog):
    """Dialog for importing videos. Probes the file and creates a single
    whole-file clip. Cut detection is a separate step."""

    import_complete = Signal(object, list)  # VideoSource, List[Clip]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Video")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Select a video file to import.")
        layout.addWidget(self._status_label)

        btn_layout = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse)
        btn_layout.addWidget(self._browse_btn)
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", "", VIDEO_FILTERS
        )
        if not paths:
            return
        self._start_import(paths[0])

    def _start_import(self, file_path: str):
        name = file_path.split('/')[-1].split(chr(92))[-1]
        self._status_label.setText(f"Probing: {name}")

        info = probe_video(file_path)
        if info is None:
            self._status_label.setText("Error: Could not read video file. Is FFmpeg installed?")
            return

        source = VideoSource(
            file_path=file_path,
            total_frames=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            codec=info.codec,
        )

        # Single clip spanning the entire file
        clip = Clip(
            source_id=source.id,
            source_in=0,
            source_out=source.total_frames - 1,
        )

        self.import_complete.emit(source, [clip])
        self.accept()
