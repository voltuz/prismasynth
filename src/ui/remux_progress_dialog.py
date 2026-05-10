"""Modal progress dialog for the timebase auto-fix run.

Owns a ``RemuxJob`` worker thread and surfaces its lifecycle to the user:
status label, indeterminate progress bar (ffmpeg ``-c copy`` doesn't expose
a meaningful percentage without parsing stderr, and a multi-GB Bluray remux
finishes in tens of seconds — a spinner conveys the right shape of wait).

Per-source completion fans out via ``source_succeeded`` so MainWindow can
relink each fixed file as it lands; partial cancels keep the already-fixed
sources fixed.
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)

from core.timebase_remuxer import RemuxJob, RemuxJobSpec


class RemuxProgressDialog(QDialog):
    """Hosts the RemuxJob worker. Buttons morph Cancel→Cancelling…→Close
    just like ``cache_thumbnails_dialog.CacheThumbnailsDialog``."""

    source_succeeded = Signal(str, str)  # (source_id, fixed_path)

    def __init__(self, jobs: List[RemuxJobSpec], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remuxing source files")
        self.setModal(True)
        self.setMinimumWidth(540)

        self._total = len(jobs)
        self._failures: list = []  # (source_id, message)

        layout = QVBoxLayout(self)

        self._status = QLabel(f"Preparing {self._total} file(s)…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        # Indeterminate while a single ffmpeg runs; flips to determinate
        # for the per-source step counter via setRange/setValue.
        self._bar.setRange(0, self._total)
        self._bar.setValue(0)
        layout.addWidget(self._bar)

        self._detail = QLabel("")
        self._detail.setStyleSheet("color: #aaa; font-size: 11px;")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._right_btn = QPushButton("Cancel")
        self._right_btn.clicked.connect(self._on_right_button)
        btn_row.addWidget(self._right_btn)
        layout.addLayout(btn_row)

        self._job = RemuxJob(jobs, parent=self)
        self._job.progress.connect(self._on_progress)
        self._job.source_done.connect(self._on_source_done)
        self._job.error.connect(self._on_error)
        self._job.cancelled.connect(self._on_cancelled)
        self._job.finished_all.connect(self._on_finished_all)
        self._job.start()

    # --- worker signals -----------------------------------------------

    def _on_progress(self, done: int, total: int, current_basename: str):
        self._bar.setValue(min(done, total))
        if current_basename:
            self._status.setText(
                f"Remuxing {done + 1} of {total}: {current_basename}")
        else:
            self._status.setText(f"Done: {done} of {total}")

    def _on_source_done(self, source_id: str, fixed_path: str):
        # Forward to MainWindow so the fixed file gets relinked
        # immediately (per-source, not end-of-batch — partial cancels
        # still leave the completed sources fixed).
        self.source_succeeded.emit(source_id, fixed_path)

    def _on_error(self, source_id: str, message: str):
        self._failures.append((source_id, message))

    def _on_cancelled(self):
        self._status.setText("Cancelled.")

    def _on_finished_all(self, succeeded: int, total: int):
        if succeeded == total:
            self._status.setText(
                f"Remuxed {succeeded} of {total} file(s) successfully.")
        else:
            self._status.setText(
                f"Remuxed {succeeded} of {total} file(s). "
                f"{total - succeeded} failed or cancelled.")
        if self._failures:
            lines = "\n".join(
                f"  • {sid}: {msg.splitlines()[-1] if msg else 'unknown'}"
                for sid, msg in self._failures)
            self._detail.setText("Errors:\n" + lines)
        self._right_btn.setText("Close")
        self._right_btn.setEnabled(True)

    # --- buttons ------------------------------------------------------

    def _on_right_button(self):
        if self._right_btn.text() == "Cancel":
            self._right_btn.setEnabled(False)
            self._right_btn.setText("Cancelling…")
            self._job.cancel()
            return
        self.accept()

    def _is_running(self) -> bool:
        return self._right_btn.text() in ("Cancel", "Cancelling…")

    def closeEvent(self, event):
        if self._is_running():
            if self._right_btn.isEnabled():
                self._on_right_button()
            event.ignore()
            return
        # Make sure the worker thread is finished before destroying it.
        if self._job.isRunning():
            self._job.wait(2000)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._is_running():
            if self._right_btn.isEnabled():
                self._on_right_button()
            event.accept()
            return
        super().keyPressEvent(event)
