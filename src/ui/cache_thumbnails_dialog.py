import logging
import time
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout,
)

from core.thumbnail_cache import BulkCacheJob, ThumbnailCache

logger = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    """Compact human-readable size (e.g. '523 KB', '12.4 MB', '1.2 GB')."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


class CacheThumbnailsDialog(QDialog):
    """Modal dialog that bakes the on-disk thumbnail cache for the first
    and last frame of every non-gap clip on the timeline. Reusable across
    multiple bakes within the same dialog session: after Done / Cancelled /
    Already-Cached the dialog returns to its idle state with Start enabled
    so the user can run another bake (e.g. with Overwrite ticked) without
    closing and reopening.
    """

    def __init__(self, thumbnail_cache: ThumbnailCache,
                 render_range: Tuple[int, int],
                 has_in_out: bool, parent=None):
        super().__init__(parent)
        self._cache = thumbnail_cache
        self._render_range = render_range
        self._has_in_out = has_in_out
        self._job: Optional[BulkCacheJob] = None
        self._signals_connected: bool = False
        self._start_time: float = 0.0
        self._last_done: int = 0

        self.setWindowTitle("Cache Thumbnails")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Caches the first and last frame of every clip on the timeline.\n"
            "Already-cached frames are skipped unless Overwrite is enabled."
        ))

        # Render-range checkbox. Disabled when there's no in/out so the
        # state can't be misinterpreted.
        self._range_check = QCheckBox("Render range only")
        self._range_check.setChecked(has_in_out)
        self._range_check.setEnabled(has_in_out)
        if not has_in_out:
            self._range_check.setToolTip(
                "Set in/out points on the timeline to enable this option."
            )
        layout.addWidget(self._range_check)

        # Overwrite checkbox — when on, the bake re-decodes frames already
        # on disk instead of skipping them. Useful when a thumbnail is
        # suspected stale or the source file changed in place.
        self._overwrite_check = QCheckBox("Overwrite existing")
        self._overwrite_check.setToolTip(
            "Re-decode and overwrite cached thumbnails for every frame in\n"
            "scope, even if they're already on disk."
        )
        layout.addWidget(self._overwrite_check)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._detail_label.setVisible(False)
        layout.addWidget(self._detail_label)

        # Buttons. Layout left → right:
        #   [Clear cached thumbnails…]   ──   [Start]   [Close]
        # The right button morphs through Close → Cancel → Cancelling…
        # → Close based on job state. Start is hidden during a run and
        # re-shown by _reset_to_idle on terminal states.
        btn_layout = QHBoxLayout()
        self._clear_btn = QPushButton("Clear cached thumbnails…")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._clear_btn)
        btn_layout.addStretch()
        self._start_btn = QPushButton("Start")
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)
        self._right_btn = QPushButton("Close")
        self._right_btn.clicked.connect(self._on_right_button)
        btn_layout.addWidget(self._right_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------ start

    def _on_start(self):
        # Drop any prior job so a delayed emit from it can't repaint us
        # with stale state when we re-bake.
        if self._job is not None and self._signals_connected:
            for sig, slot in (
                (self._job.progress, self._on_progress),
                (self._job.finished, self._on_finished),
                (self._job.cancelled, self._on_cancelled),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._signals_connected = False
        self._job = None

        force_overwrite = self._overwrite_check.isChecked()
        if self._range_check.isChecked():
            render_range: Optional[Tuple[int, int]] = self._render_range
        else:
            render_range = None
        self._job = self._cache.start_bulk_cache(
            render_range, force_overwrite=force_overwrite)

        # Empty case: nothing to do. Don't even start the job's thread —
        # just tell the user and return to idle so they can try again
        # (e.g. with Overwrite ticked).
        if self._job.total == 0:
            self._detail_label.setVisible(True)
            self._detail_label.setText("All thumbnails already cached.")
            self._progress.setVisible(False)
            self._reset_to_idle()
            return

        self._job.progress.connect(
            self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._job.finished.connect(
            self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._job.cancelled.connect(
            self._on_cancelled, Qt.ConnectionType.QueuedConnection)
        self._signals_connected = True

        self._set_running_state(True)
        self._progress.setValue(0)
        self._detail_label.setVisible(True)
        self._detail_label.setText(f"Caching 0 / {self._job.total}…")
        self._start_time = time.monotonic()
        self._last_done = 0
        self._job.start()

    # ----------------------------------------------------------------- clear

    def _on_clear(self):
        source_ids = list(self._cache._sources.keys())
        n, total_bytes = self._cache.disk_thumbnail_stats(source_ids)
        if n == 0:
            self._detail_label.setVisible(True)
            self._detail_label.setText(
                "Nothing to clear — no cached thumbnails for the loaded "
                "sources."
            )
            return
        n_sources = len(source_ids)
        msg = (
            f"Delete {n:,} cached thumbnail(s) "
            f"({_fmt_bytes(total_bytes)}) for the {n_sources} source(s) "
            f"loaded in this project?\n\n"
            f"Other projects' cached thumbnails are not affected."
        )
        choice = QMessageBox.question(
            self, "Clear thumbnail cache", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        cleared_n, cleared_b = self._cache.clear_disk_thumbnails(source_ids)
        self._progress.setVisible(False)
        self._detail_label.setVisible(True)
        self._detail_label.setText(
            f"Cleared {cleared_n:,} thumbnail(s) ({_fmt_bytes(cleared_b)})."
        )

    # ---------------------------------------------------------------- updates

    def _on_progress(self, done: int, total: int):
        self._last_done = done
        if total > 0:
            self._progress.setValue(int(done * 100 / total))
        elapsed = time.monotonic() - self._start_time
        eta_str = ""
        if done > 0 and elapsed > 1.0:
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            if remaining >= 60:
                eta_str = (f"  ~{int(remaining) // 60}m "
                           f"{int(remaining) % 60:02d}s left")
            else:
                eta_str = f"  ~{int(remaining)}s left"
        self._detail_label.setText(f"Caching {done:,} / {total:,}{eta_str}")

    def _on_finished(self):
        elapsed = time.monotonic() - self._start_time
        total = self._job.total if self._job else 0
        if total > 0:
            self._progress.setValue(100)
            self._detail_label.setText(
                f"Done — {total:,} thumbnails cached in {int(elapsed)}s"
            )
        self._reset_to_idle()

    def _on_cancelled(self):
        self._detail_label.setText(
            f"Cancelled — {self._last_done:,} thumbnails cached before "
            f"stopping."
        )
        self._reset_to_idle()

    # ---------------------------------------------------------------- buttons

    def _on_right_button(self):
        # State is implied by the button's current text:
        # - "Cancel"  → job is running, ask it to stop and disable until
        #               the job confirms with cancelled / finished.
        # - "Close"   → just close.
        if self._right_btn.text() == "Cancel":
            self._right_btn.setEnabled(False)
            self._right_btn.setText("Cancelling…")
            if self._job is not None:
                self._job.cancel()
            return
        self.accept()

    # ----------------------------------------------------------------- state

    def _set_running_state(self, running: bool):
        """Toggle every control's enable/visibility for the running phase."""
        self._start_btn.setVisible(not running)
        self._range_check.setEnabled(
            (not running) and self._has_in_out)
        self._overwrite_check.setEnabled(not running)
        self._clear_btn.setEnabled(not running)
        if running:
            self._progress.setVisible(True)
            self._right_btn.setText("Cancel")
            self._right_btn.setEnabled(True)
        else:
            self._right_btn.setText("Close")
            self._right_btn.setEnabled(True)

    def _reset_to_idle(self):
        """Return the dialog to its initial controls so the user can
        immediately re-bake (e.g. with Overwrite ticked) without closing.
        Keeps the detail label visible so the last result stays readable."""
        self._set_running_state(False)
        # Hide the progress bar between runs so a stale 100% / 0% value
        # doesn't sit there confusingly. The detail label keeps the
        # outcome ("Done — N…", "Cancelled — N…", "All cached") visible.
        self._progress.setVisible(False)

    # ------------------------------------------------------------------ close

    def _is_running(self) -> bool:
        # Running iff the right button is in the cancel phase.
        return self._right_btn.text() in ("Cancel", "Cancelling…")

    def closeEvent(self, event):
        if self._is_running():
            # Don't let X / Esc close while the job is still running.
            # Trigger cancel (idempotent) and wait for confirmation.
            if self._right_btn.isEnabled():
                self._on_right_button()
            event.ignore()
            return
        super().closeEvent(event)

    def keyPressEvent(self, event):
        # Standard Qt would let Esc reject the dialog immediately; intercept
        # while the job is running so it routes through the same cancel path.
        if event.key() == Qt.Key.Key_Escape and self._is_running():
            if self._right_btn.isEnabled():
                self._on_right_button()
            event.accept()
            return
        super().keyPressEvent(event)
