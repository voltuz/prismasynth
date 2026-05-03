"""Non-modal source-info window.

Opened from the Media Panel via double-click. A single instance is reused
across the session — double-clicking another source rebinds this window
instead of stacking N windows. Closing it just hides it; the next double-click
brings it back.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from core.video_source import VideoSource
from core.source_thumbnail import cache_path_for


class SourceInfoDialog(QDialog):
    """Non-modal source info window. Reuse a single instance via show_for()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setMinimumWidth(420)
        self.setWindowTitle("Source Info")
        # Stay on top of the parent without blocking it
        self.setWindowFlag(Qt.WindowType.Tool, True)

        layout = QVBoxLayout(self)

        self._thumb = QLabel()
        self._thumb.setFixedSize(320, 180)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background: #1e1e1e; border: 1px solid #444;")
        layout.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignCenter)

        self._title = QLabel("")
        self._title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ddd;")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        form = QFormLayout()
        self._path_label = QLabel("-")
        self._path_label.setWordWrap(True)
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._codec_label = QLabel("-")
        self._resolution_label = QLabel("-")
        self._fps_label = QLabel("-")
        self._duration_label = QLabel("-")
        self._frames_label = QLabel("-")
        self._audio_label = QLabel("-")
        for lbl in (self._codec_label, self._resolution_label, self._fps_label,
                    self._duration_label, self._frames_label, self._audio_label,
                    self._path_label):
            lbl.setStyleSheet("color: #ccc;")

        form.addRow("File:", self._path_label)
        form.addRow("Codec:", self._codec_label)
        form.addRow("Resolution:", self._resolution_label)
        form.addRow("FPS:", self._fps_label)
        form.addRow("Duration:", self._duration_label)
        form.addRow("Frames:", self._frames_label)
        form.addRow("Audio:", self._audio_label)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # --- Public ---

    def show_for(self, source: VideoSource):
        """Bind to a source and bring the window forward."""
        name = Path(source.file_path).name
        self.setWindowTitle(f"Source Info — {name}")
        self._title.setText(name)
        self._path_label.setText(source.file_path)
        self._codec_label.setText(source.codec or "-")
        self._resolution_label.setText(f"{source.width} × {source.height}")
        self._fps_label.setText(f"{source.fps:.3f} fps")

        secs_total = int(source.duration_seconds)
        h, rem = divmod(secs_total, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        self._duration_label.setText(time_str)
        self._frames_label.setText(f"{source.total_frames:,}")

        audio = source.format_audio()
        self._audio_label.setText(audio)
        self._audio_label.setStyleSheet(
            "color: #e8a735;" if audio == "none" else "color: #ccc;"
        )

        # Thumbnail (best-effort — extract_thumbnail is called at import time
        # so the cache should already be warm)
        thumb_path = cache_path_for(source)
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            pix = QPixmap(str(thumb_path))
            if not pix.isNull():
                self._thumb.setPixmap(pix.scaled(
                    self._thumb.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            else:
                self._thumb.setText("(no preview)")
        else:
            self._thumb.setText("(no preview)")

        self.show()
        self.raise_()
        self.activateWindow()
