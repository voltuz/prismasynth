import logging
import time
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QDoubleSpinBox, QFormLayout, QComboBox,
)
from PySide6.QtCore import Signal

from core.video_source import VideoSource
from core.scene_detector import SceneDetector, DEFAULT_THRESHOLD, Detector
from core.omnishotcut_runner import is_setup_complete, default_checkpoint_path
from core.clip import Clip

logger = logging.getLogger(__name__)


class DetectDialog(QDialog):
    """Dialog for running cut detection on all clips currently on the timeline."""

    detection_complete = Signal(dict)  # {clip_id: [Clip, ...]}

    _PHASE_STEPS = {"Decoding": 1, "Analyzing": 2}

    def __init__(self, segments: list, sources: dict,
                 in_out_limited: bool = False, parent=None):
        """segments: list of (source_id, source_in, source_out, clip_id).
        sources: {source_id: VideoSource}."""
        super().__init__(parent)
        self._segments = segments
        self._sources = sources
        self.setWindowTitle("Detect Cuts")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Summary info
        total_frames = sum(seg[2] - seg[1] + 1 for seg in segments)
        n_sources = len({seg[0] for seg in segments})
        n_clips = len(segments)

        if n_sources == 1:
            source = sources[segments[0][0]]
            name = source.file_path.split('/')[-1].split(chr(92))[-1]
            info_text = (f"{name}\n"
                         f"{source.width}x{source.height}, {source.fps:.2f} fps, "
                         f"{total_frames:,} frames across {n_clips} clip(s)")
        else:
            first_source = sources[segments[0][0]]
            info_text = (f"{n_sources} sources, {n_clips} clip(s)\n"
                         f"{first_source.width}x{first_source.height}, "
                         f"{first_source.fps:.2f} fps, "
                         f"{total_frames:,} total frames")

        layout.addWidget(QLabel(info_text))

        if in_out_limited:
            warn = QLabel("Detection limited to in/out render range.")
            warn.setStyleSheet("color: #e8a735; font-size: 11px;")
            layout.addWidget(warn)

        # Detector + threshold form
        form = QFormLayout()

        self._detector_combo = QComboBox()
        self._detector_combo.addItem("TransNetV2 (fast, hard cuts)", Detector.TRANSNETV2)
        self._detector_combo.addItem(
            "OmniShotCut (slower, detects dissolves/fades/wipes)", Detector.OMNISHOTCUT,
        )
        self._detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        form.addRow("Detector:", self._detector_combo)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.05, 0.95)
        self._threshold_spin.setValue(DEFAULT_THRESHOLD)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setToolTip(
            "TransNetV2 confidence threshold (0-1). "
            "Higher = fewer cuts (only high-confidence). Lower = more sensitive."
        )
        self._threshold_label = QLabel("Cut Detection Sensitivity:")
        form.addRow(self._threshold_label, self._threshold_spin)
        layout.addLayout(form)

        self._omnishotcut_status = QLabel("")
        self._omnishotcut_status.setStyleSheet("color: #e8a735; font-size: 11px;")
        self._omnishotcut_status.setVisible(False)
        layout.addWidget(self._omnishotcut_status)

        # Step indicator (above the progress bar) — "Step 1 of 2 — Decoding" etc.
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("font-weight: bold; color: #ddd;")
        self._step_label.setVisible(False)
        layout.addWidget(self._step_label)

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
        self._setup_btn = QPushButton("Set up OmniShotCut")
        self._setup_btn.clicked.connect(self._run_setup)
        self._setup_btn.setVisible(False)
        btn_layout.addWidget(self._setup_btn)
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        self._worker: Optional[SceneDetector] = None
        self._start_time: float = 0
        self._phase_start_time: float = 0

        # Initialise UI state for the default detector
        self._on_detector_changed()

    def _selected_detector(self) -> Detector:
        return self._detector_combo.currentData() or Detector.TRANSNETV2

    def _on_detector_changed(self, *_):
        """Swap visible widgets based on the selected detector."""
        det = self._selected_detector()
        is_transnet = (det == Detector.TRANSNETV2)
        # Threshold only makes sense for TransNetV2
        self._threshold_spin.setVisible(is_transnet)
        self._threshold_label.setVisible(is_transnet)

        if is_transnet:
            self._omnishotcut_status.setVisible(False)
            self._setup_btn.setVisible(False)
            self._start_btn.setVisible(True)
            self._start_btn.setEnabled(True)
        else:
            if is_setup_complete():
                self._omnishotcut_status.setText(
                    "OmniShotCut ready. Detection runs in a sidecar process — first segment "
                    "may take 5-15 s while the model loads."
                )
                self._omnishotcut_status.setStyleSheet("color: #6cba7e; font-size: 11px;")
                self._omnishotcut_status.setVisible(True)
                self._setup_btn.setVisible(False)
                self._start_btn.setVisible(True)
                self._start_btn.setEnabled(True)
            else:
                self._omnishotcut_status.setText(
                    "OmniShotCut is not installed. Setting up downloads ~3 GB and may take "
                    "5-15 minutes."
                )
                self._omnishotcut_status.setStyleSheet("color: #e8a735; font-size: 11px;")
                self._omnishotcut_status.setVisible(True)
                self._setup_btn.setVisible(True)
                self._start_btn.setVisible(False)

    def _run_setup(self):
        """Open the setup dialog. On success, refresh the detector state."""
        # Lazy import — avoids circular when this module loads
        from ui.omnishotcut_setup_dialog import OmnishotcutSetupDialog
        dlg = OmnishotcutSetupDialog(self)
        dlg.exec()
        self._on_detector_changed()

    def _start_detection(self):
        det = self._selected_detector()
        if det == Detector.OMNISHOTCUT and not is_setup_complete():
            return  # button shouldn't be reachable, but guard anyway

        self._start_btn.setEnabled(False)
        self._threshold_spin.setEnabled(False)
        self._detector_combo.setEnabled(False)
        self._step_label.setVisible(True)
        self._step_label.setText("Starting...")
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._detail_label.setVisible(True)
        self._detail_label.setText("Starting...")
        self._start_time = time.monotonic()
        self._phase_start_time = self._start_time

        threshold = self._threshold_spin.value()
        checkpoint = str(default_checkpoint_path()) if det == Detector.OMNISHOTCUT else None
        self._worker = SceneDetector(
            self._segments, self._sources,
            threshold=threshold,
            detector=det,
            omnishotcut_checkpoint=checkpoint,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.detail_progress.connect(self._on_detail_progress)
        self._worker.phase_changed.connect(self._on_phase_changed)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int):
        self._progress.setValue(pct)

    def _on_phase_changed(self, phase: str):
        """Reset the phase timer when switching stages (e.g. decode → inference)."""
        self._phase_start_time = time.monotonic()
        self._detail_label.setText(f"{phase}...")
        step = self._PHASE_STEPS.get(phase)
        if step:
            self._step_label.setText(f"Step {step} of 2 — {phase}")
        else:
            # Unknown phases (e.g. OmniShotCut "Loading model") just show the name
            self._step_label.setText(phase)

    def _on_detail_progress(self, frames_done: int, total_frames: int, phase: str):
        elapsed = time.monotonic() - self._start_time
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

    def _on_finished(self, results: dict):
        elapsed = time.monotonic() - self._start_time
        total_clips = sum(len(v) for v in results.values())
        self._progress.setValue(100)
        self._step_label.setText("Done")
        self._detail_label.setText(
            f"Done — {total_clips} clips from {len(results)} segment(s) in {int(elapsed)}s"
        )
        self.detection_complete.emit(results)
        self.accept()

    def _on_error(self, msg: str):
        self._threshold_spin.setEnabled(True)
        self._detector_combo.setEnabled(True)
        self._progress.setVisible(False)
        self._step_label.setVisible(False)
        self._detail_label.setText(f"Error: {msg}")
        # Re-evaluate which buttons should be visible (handles e.g. OmniShotCut
        # error → user wants to switch back to TransNetV2 or re-run setup).
        self._on_detector_changed()
        logger.error("Detection failed: %s", msg)

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            # Wait longer — GPU inference windows can take a moment to finish
            if not self._worker.wait(5000):
                logger.warning("Detection thread did not stop in 5s, terminating")
                self._worker.terminate()
                self._worker.wait(2000)
        self.reject()

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
