import logging
import time
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QDoubleSpinBox, QFormLayout,
)
from PySide6.QtCore import Signal

from core.video_source import VideoSource
from core.scene_detector import SceneDetector, DEFAULT_THRESHOLD
from core.clip import Clip

logger = logging.getLogger(__name__)


class DetectDialog(QDialog):
    """Dialog for running cut detection on an already-imported video source."""

    detection_complete = Signal(str, list)  # source_id, List[Clip]

    def __init__(self, source: VideoSource, parent=None):
        super().__init__(parent)
        self._source = source
        self.setWindowTitle("Detect Cuts")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        name = source.file_path.split('/')[-1].split(chr(92))[-1]
        layout.addWidget(QLabel(
            f"{name}\n"
            f"{source.width}x{source.height}, {source.fps:.2f} fps, "
            f"{source.total_frames:,} frames"
        ))

        # Threshold setting
        form = QFormLayout()
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.05, 0.95)
        self._threshold_spin.setValue(DEFAULT_THRESHOLD)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setToolTip(
            "TransNetV2 confidence threshold (0-1). "
            "Higher = fewer cuts (only high-confidence). Lower = more sensitive."
        )
        form.addRow("Cut Detection Sensitivity:", self._threshold_spin)
        layout.addLayout(form)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Detail label: frame counter + ETA
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._detail_label.setVisible(False)
        layout.addWidget(self._detail_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("Detect")
        self._start_btn.clicked.connect(self._start_detection)
        btn_layout.addWidget(self._start_btn)
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        self._detector: Optional[SceneDetector] = None
        self._start_time: float = 0
        self._phase_start_time: float = 0

    def _start_detection(self):
        self._start_btn.setEnabled(False)
        self._threshold_spin.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._detail_label.setVisible(True)
        self._detail_label.setText("Starting...")
        self._start_time = time.monotonic()
        self._phase_start_time = self._start_time

        threshold = self._threshold_spin.value()
        self._detector = SceneDetector(self._source, threshold=threshold)
        self._detector.progress.connect(self._on_progress)
        self._detector.detail_progress.connect(self._on_detail_progress)
        self._detector.phase_changed.connect(self._on_phase_changed)
        self._detector.finished.connect(self._on_finished)
        self._detector.error.connect(self._on_error)
        self._detector.start()

    def _on_progress(self, pct: int):
        self._progress.setValue(pct)

    def _on_phase_changed(self, phase: str):
        """Reset the phase timer when switching stages (decode → inference)."""
        self._phase_start_time = time.monotonic()
        self._detail_label.setText(f"{phase}...")

    def _on_detail_progress(self, frames_done: int, total_frames: int, phase: str):
        elapsed = time.monotonic() - self._phase_start_time
        eta_str = ""
        if frames_done > 0 and elapsed > 1.0:
            rate = frames_done / elapsed
            remaining = (total_frames - frames_done) / rate
            if remaining >= 60:
                eta_str = f"  ~{int(remaining) // 60}m {int(remaining) % 60:02d}s left"
            else:
                eta_str = f"  ~{int(remaining)}s left"

        self._detail_label.setText(
            f"{phase}: {frames_done:,} / {total_frames:,} frames{eta_str}"
        )

    def _on_finished(self, clips: List[Clip]):
        elapsed = time.monotonic() - self._start_time
        self._progress.setValue(100)
        self._detail_label.setText(
            f"Done — {len(clips)} cuts found in {int(elapsed)}s"
        )
        self.detection_complete.emit(self._source.id, clips)
        self.accept()

    def _on_error(self, msg: str):
        self._start_btn.setEnabled(True)
        self._threshold_spin.setEnabled(True)
        self._progress.setVisible(False)
        self._detail_label.setText(f"Error: {msg}")
        logger.error("Detection failed: %s", msg)

    def _cancel(self):
        if self._detector and self._detector.isRunning():
            self._detector.cancel()
            # Wait longer — GPU inference windows can take a moment to finish
            if not self._detector.wait(5000):
                logger.warning("Detection thread did not stop in 5s, terminating")
                self._detector.terminate()
                self._detector.wait(2000)
        self.reject()

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
