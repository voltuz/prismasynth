import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal

from core.video_source import VideoSource
from utils.ffprobe import probe_video

logger = logging.getLogger(__name__)

VIDEO_FILTERS = "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v *.ts *.mxf);;All Files (*)"


class ImportDialog(QDialog):
    """Dialog for importing one or more videos. Probes all selected files,
    validates that they share the same resolution and FPS (and match existing
    sources if any), and adds them to the Media Pool. Sources do NOT
    automatically create timeline clips — the user drags them onto the
    timeline from the Media Pool when ready. Cut detection is also separate."""

    # Emits list of VideoSource objects to add to the media pool.
    import_complete = Signal(list)  # List[VideoSource]

    def __init__(self, ref_width: int = 0, ref_height: int = 0,
                 ref_fps: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Video")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._ref_width = ref_width
        self._ref_height = ref_height
        self._ref_fps = ref_fps

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Select video file(s) to import.")
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
        self._start_import(paths)

    def _start_import(self, file_paths: list):
        self._status_label.setText(f"Probing {len(file_paths)} file(s)...")

        # Probe all files
        probed = []
        for path in file_paths:
            info = probe_video(path)
            if info is None:
                name = path.split('/')[-1].split(chr(92))[-1]
                QMessageBox.critical(
                    self, "Import Error",
                    f"Could not read video file:\n{name}\n\n"
                    "Is FFmpeg installed and on PATH?"
                )
                self._status_label.setText("Select video file(s) to import.")
                return
            probed.append((path, info))

        # Determine reference resolution/fps: use existing timeline if set,
        # otherwise use first file
        if self._ref_width > 0:
            ref_w, ref_h, ref_fps = self._ref_width, self._ref_height, self._ref_fps
            ref_label = "timeline"
        else:
            first_path, first_info = probed[0]
            ref_w, ref_h, ref_fps = first_info.width, first_info.height, first_info.fps
            ref_label = first_path.split('/')[-1].split(chr(92))[-1]

        # Validate all files match reference
        for path, info in probed:
            name = path.split('/')[-1].split(chr(92))[-1]
            if info.width != ref_w or info.height != ref_h:
                QMessageBox.critical(
                    self, "Import Error",
                    f"Resolution mismatch — batch rejected.\n\n"
                    f"{name} is {info.width}x{info.height}\n"
                    f"Expected {ref_w}x{ref_h} (from {ref_label})"
                )
                self._status_label.setText("Select video file(s) to import.")
                return
            if abs(info.fps - ref_fps) > 0.02:
                QMessageBox.critical(
                    self, "Import Error",
                    f"FPS mismatch — batch rejected.\n\n"
                    f"{name} is {info.fps:.3f} fps\n"
                    f"Expected {ref_fps:.3f} fps (from {ref_label})"
                )
                self._status_label.setText("Select video file(s) to import.")
                return

        # Build sources only — clips are created later when the user drags
        # from the media pool to the timeline.
        sources = []
        for path, info in probed:
            sources.append(VideoSource(
                file_path=path,
                total_frames=info.total_frames,
                fps=info.fps,
                width=info.width,
                height=info.height,
                codec=info.codec,
                audio_codec=info.audio_codec,
                audio_sample_rate=info.audio_sample_rate,
                audio_channels=info.audio_channels,
            ))

        self.import_complete.emit(sources)
        self.accept()
