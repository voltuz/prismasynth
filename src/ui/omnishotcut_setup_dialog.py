"""Modal dialog that runs scripts/setup_omnishotcut.py and tails its output.

Used by DetectDialog when the user picks OmniShotCut but the sidecar venv +
checkpoint aren't installed yet.
"""

import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from core.omnishotcut_runner import is_setup_complete
from core.ui_scale import ui_scale

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_omnishotcut.py"


class _SetupWorker(QThread):
    """Spawns setup_omnishotcut.py and streams its stdout line-by-line."""

    line = Signal(str)
    finished_ok = Signal(bool)  # True on success, False on failure or cancel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self):
        try:
            cmd = [sys.executable, str(SETUP_SCRIPT)]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(REPO_ROOT),
            )
            for raw in self._proc.stdout:
                if self._cancelled:
                    break
                self.line.emit(raw.rstrip("\n"))
            self._proc.wait()
            ok = (not self._cancelled) and self._proc.returncode == 0 and is_setup_complete()
            self.finished_ok.emit(ok)
        except Exception as e:
            self.line.emit(f"[setup-dialog] worker error: {e}")
            self.finished_ok.emit(False)


class OmnishotcutSetupDialog(QDialog):
    """Streams scripts/setup_omnishotcut.py output. Returns Accepted on success."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up OmniShotCut")
        _s = ui_scale()
        self.setMinimumSize(_s.px(700), _s.px(500))
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Installing OmniShotCut into venv-omnishotcut/. This downloads ~3 GB\n"
            "(uv, Python 3.10, torch + CUDA wheels, OmniShotCut model checkpoint)\n"
            "and may take 5-15 minutes on first run."
        ))

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        f = QFont("Consolas", ui_scale().font_pt(9))
        f.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(f)
        self._log.setStyleSheet("QPlainTextEdit { background: #1e1e1e; color: #ddd; }")
        layout.addWidget(self._log, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        self._worker = _SetupWorker(self)
        self._worker.line.connect(self._on_line, Qt.ConnectionType.QueuedConnection)
        self._worker.finished_ok.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.start()

    def _on_line(self, text: str):
        self._log.appendPlainText(text)
        bar = self._log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_finished(self, ok: bool):
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        if ok:
            self._log.appendPlainText("\n[setup-dialog] Setup complete. You can now run OmniShotCut detection.")
            self._cancel_btn.clicked.connect(self.accept)
        else:
            self._log.appendPlainText("\n[setup-dialog] Setup did not complete successfully.")
            self._cancel_btn.clicked.connect(self.reject)
        self._cancel_btn.setEnabled(True)

    def _on_cancel(self):
        if self._worker.isRunning():
            self._log.appendPlainText("\n[setup-dialog] Cancelling...")
            self._cancel_btn.setEnabled(False)
            self._worker.cancel()
            if not self._worker.wait(5000):
                logger.warning("Setup worker didn't stop in 5s; terminating")
                self._worker.terminate()
                self._worker.wait(2000)
        self.reject()

    def closeEvent(self, event):
        self._on_cancel()
        super().closeEvent(event)
