"""System Performance Check dialog.

Surfaces whether every GPU/codec/filter piece of the pipeline (ffmpeg,
NVDEC, NVENC, OpenCL tonemap, scale_cuda, mpv hwdec, OmniShotCut sidecar)
is actually working, and for each row that isn't, what to do about it.

Probes live in utils.diagnostics; this file is just the Qt shell.

Probes run on a background QThread so the dialog appears immediately
with "Running…" placeholders and each row flips to its real status as
the probe completes.
"""

import logging

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from core.ui_scale import ui_scale
from utils.diagnostics import (
    FAIL, PASS, SKIP, WARN, CheckResult, check_nvdec_runtime_from_value,
    format_as_text, probe_names, run_probe,
)

logger = logging.getLogger(__name__)


_STATUS_COLOR = {
    PASS: QColor("#5fbf5f"),   # green
    WARN: QColor("#d6a93b"),   # amber
    FAIL: QColor("#d65555"),   # red
    SKIP: QColor("#888888"),   # grey
}
_RUNNING_COLOR = QColor("#888888")
_NVDEC_RUNTIME_ROW_NAME = "NVDEC (runtime engaged)"


class _ProbeWorker(QThread):
    """Runs the probe registry off the UI thread, one probe at a time."""

    result_ready = Signal(int, object)   # row index, CheckResult
    all_finished = Signal()

    def __init__(self, prebuilt: dict, parent=None):
        super().__init__(parent)
        # {row_index: CheckResult} — entries override calls to run_probe(i).
        # Used for probes that must execute on the UI thread (mpv).
        self._prebuilt = prebuilt

    def run(self):
        n = len(probe_names())
        for i in range(n):
            if i in self._prebuilt:
                r = self._prebuilt[i]
            else:
                r = run_probe(i)
            self.result_ready.emit(i, r)
        self.all_finished.emit()


class DiagnosticsDialog(QDialog):
    """Modal pass/warn/fail report with suggested fixes and a Refresh button.

    Probes run on a background QThread; rows show "Running…" until each
    probe completes, then flip to PASS/WARN/FAIL/SKIP. Refresh/Copy/Close
    are disabled while a run is in flight to keep the lifecycle simple.
    """

    def __init__(self, parent=None, preview_widget=None):
        super().__init__(parent)
        self.setWindowTitle("System Performance Check")
        self.setModal(True)
        _s = ui_scale()
        self.setMinimumWidth(_s.px(720))
        self.setMinimumHeight(_s.px(420))

        self._preview_widget = preview_widget
        self._last_results: list = [None] * len(probe_names())
        self._worker: _ProbeWorker | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Static checks confirm that every GPU / codec / filter piece "
            "PrismaSynth relies on for fast playback, scene detection, "
            "and export is working. Failures include a suggested fix.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Check", "Status", "Detail & Fix"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setWordWrap(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideNone)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._start_run)
        bottom.addWidget(self._refresh_btn)
        self._copy_btn = QPushButton("Copy to Clipboard")
        self._copy_btn.clicked.connect(self._copy_to_clipboard)
        bottom.addWidget(self._copy_btn)
        self._close_btn = QPushButton("Close")
        self._close_btn.setDefault(True)
        self._close_btn.clicked.connect(self.accept)
        bottom.addWidget(self._close_btn)
        layout.addLayout(bottom)

        self._start_run()

    def _start_run(self):
        """Reset the table to 'Running…' placeholders, disable buttons,
        and kick off a worker thread that runs every probe."""
        names = probe_names()

        # Populate placeholder rows.
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(names))
            for row, name in enumerate(names):
                name_item = QTableWidgetItem(name)
                status_item = QTableWidgetItem("Running…")
                status_item.setForeground(QBrush(_RUNNING_COLOR))
                font = status_item.font()
                font.setItalic(True)
                status_item.setFont(font)
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                detail_item = QTableWidgetItem("")
                self._table.setItem(row, 0, name_item)
                self._table.setItem(row, 1, status_item)
                self._table.setItem(row, 2, detail_item)
        finally:
            self._table.setUpdatesEnabled(True)
        QTimer.singleShot(0, self._table.resizeRowsToContents)

        self._last_results = [None] * len(names)

        # Precompute the mpv-touching probe on the UI thread.
        prebuilt: dict[int, CheckResult] = {}
        try:
            nvdec_row = names.index(_NVDEC_RUNTIME_ROW_NAME)
        except ValueError:
            nvdec_row = -1
        if nvdec_row >= 0:
            prebuilt[nvdec_row] = self._build_nvdec_runtime_result()

        self._set_buttons_enabled(False)

        self._worker = _ProbeWorker(prebuilt, parent=self)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _build_nvdec_runtime_result(self) -> CheckResult:
        """Read mpv's hwdec-current on the UI thread and synthesise the
        NVDEC-runtime row. mpv property access isn't thread-safe."""
        if self._preview_widget is None:
            return CheckResult(
                _NVDEC_RUNTIME_ROW_NAME, SKIP,
                "No preview widget available — open the main window first.",
            )
        try:
            current = self._preview_widget.get_hwdec_current()
        except Exception as e:
            return CheckResult(
                _NVDEC_RUNTIME_ROW_NAME, FAIL,
                f"Reading mpv hwdec-current raised: {e}",
                None,
            )
        return check_nvdec_runtime_from_value(current)

    def _on_result_ready(self, row: int, r: CheckResult):
        self._last_results[row] = r

        self._table.setUpdatesEnabled(False)
        try:
            name_item = QTableWidgetItem(r.name)
            status_item = QTableWidgetItem(r.status)
            status_item.setForeground(QBrush(
                _STATUS_COLOR.get(r.status, QColor("white"))))
            font = status_item.font()
            font.setBold(True)
            status_item.setFont(font)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            detail_text = r.detail
            if r.fix:
                detail_text = f"{r.detail}\nFix: {r.fix}"
            detail_item = QTableWidgetItem(detail_text)
            detail_item.setToolTip(detail_text)

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, status_item)
            self._table.setItem(row, 2, detail_item)
        finally:
            self._table.setUpdatesEnabled(True)
        QTimer.singleShot(0, self._table.resizeRowsToContents)

    def _on_all_finished(self):
        self._set_buttons_enabled(True)
        self._worker = None

    def _set_buttons_enabled(self, enabled: bool):
        self._refresh_btn.setEnabled(enabled)
        self._copy_btn.setEnabled(enabled)
        self._close_btn.setEnabled(enabled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-wrap the Detail column when the dialog width changes. Single-shot
        # again so we measure after Qt has propagated the new size.
        QTimer.singleShot(0, self._table.resizeRowsToContents)

    def reject(self):
        # Esc / window-close: refuse while probes are running so the worker
        # never outlives its parent QDialog.
        if self._worker is not None and self._worker.isRunning():
            return
        super().reject()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)

    def _copy_to_clipboard(self):
        results = [r for r in self._last_results if r is not None]
        text = format_as_text(results)
        QGuiApplication.clipboard().setText(text)
