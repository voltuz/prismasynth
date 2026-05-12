"""Project Versions dialog — list, snapshot, restore, delete.

Surfaces every snapshot stored in `<project>.psynth.versions/`. Each row
shows when it was taken, why (autosave / manual / pre-op), an optional
label, the clip and source count at that point, and the file size.

Restore is destructive on the *live* project state but reversible: a
fresh `pre_restore` snapshot is always taken before swapping in the
chosen version. The actual swap is delegated back to MainWindow via the
`restore_requested(filename)` signal because state-loading touches
readers, proxies, thumbnails, and the timeline model — all owned there.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView, QInputDialog,
)

from core.ui_scale import ui_scale
from core.project_versions import ProjectVersionStore, VersionEntry

logger = logging.getLogger(__name__)


_TRIGGER_LABELS = {
    "autosave":          "Autosave",
    "manual":            "Manual",
    "pre_detect_cuts":   "Before Detect Cuts",
    "pre_multi_delete":  "Before Multi-Delete",
    "pre_group_delete":  "Before Group Delete",
    "pre_source_removal": "Before Source Remove",
    "pre_restore":       "Before Restore",
}


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    val = float(num_bytes) / 1024.0
    for unit in ("KB", "MB", "GB", "TB"):
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


def _format_timestamp(iso: str) -> str:
    # iso is "YYYY-MM-DDTHH:MM:SS" — keep the date+time, just swap the T.
    return iso.replace("T", " ")


class VersionsDialog(QDialog):
    """Modal version browser tied to a single project."""

    # Emitted when the user confirms restore. MainWindow handles the actual
    # state swap (including the pre_restore snapshot) so this dialog stays
    # decoupled from reader/proxy/thumbnail lifecycle.
    restore_requested = Signal(str)  # filename

    def __init__(self, store: ProjectVersionStore, parent=None):
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("Project Versions")
        self.setModal(True)
        _s = ui_scale()
        self.setMinimumWidth(_s.px(680))
        self.setMinimumHeight(_s.px(380))

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Snapshots of this project's history. Versions are taken on every "
            "autosave, before risky operations, and whenever you click "
            "Snapshot Now. Restoring an old version takes a fresh snapshot of "
            "the current state first, so the rollback itself is reversible.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Timestamp", "Trigger", "Label", "Clips", "Sources", "Size"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.doubleClicked.connect(lambda _i: self._on_restore())
        layout.addWidget(self._table)

        self._summary = QLabel("")
        bottom = QHBoxLayout()
        bottom.addWidget(self._summary)
        bottom.addStretch()

        self._snapshot_btn = QPushButton("Snapshot Now…")
        self._snapshot_btn.clicked.connect(self._on_snapshot_now)
        bottom.addWidget(self._snapshot_btn)

        self._restore_btn = QPushButton("Restore")
        self._restore_btn.clicked.connect(self._on_restore)
        bottom.addWidget(self._restore_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        bottom.addWidget(self._delete_btn)

        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        self._refresh()

    # ------------------------------------------------------------------

    def _refresh(self):
        entries = self._store.list_versions()
        self._table.setRowCount(len(entries))
        total = 0
        for row, e in enumerate(entries):
            self._set_row(row, e)
            total += e.size_bytes
        self._summary.setText(
            f"{len(entries)} version(s), {_format_size(total)} on disk")
        self._update_buttons()

    def _set_row(self, row: int, e: VersionEntry):
        items = [
            _format_timestamp(e.timestamp),
            _TRIGGER_LABELS.get(e.trigger, e.trigger),
            e.label or "",
            str(e.clip_count),
            str(e.source_count),
            _format_size(e.size_bytes),
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            if col >= 3:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            # Stash the filename on column 0 for retrieval on action.
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, e.filename)
            self._table.setItem(row, col, item)

    def _selected_filename(self) -> Optional[str]:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), 0)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _update_buttons(self):
        has_sel = self._selected_filename() is not None
        self._restore_btn.setEnabled(has_sel)
        self._delete_btn.setEnabled(has_sel)

    # ------------------------------------------------------------------

    def _on_snapshot_now(self):
        label, ok = QInputDialog.getText(
            self, "Snapshot Project",
            "Optional label for this snapshot:")
        if not ok:
            return
        entry = self._store.create(trigger="manual",
                                   label=label.strip() or None)
        if entry is None:
            QMessageBox.warning(
                self, "Snapshot Failed",
                "Could not create the snapshot. Make sure the project has "
                "been saved at least once.")
            return
        self._refresh()

    def _on_restore(self):
        fname = self._selected_filename()
        if not fname:
            return
        ts = _format_timestamp(self._table.item(
            self._table.currentRow(), 0).text())
        ret = QMessageBox.warning(
            self, "Restore Version",
            f"Replace the current project with the version from {ts}?\n\n"
            "A snapshot of the current state will be taken first, so this "
            "rollback is reversible.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        # Hand off to MainWindow — it owns reader / proxy / thumbnail state.
        self.restore_requested.emit(fname)
        self.accept()

    def _on_delete(self):
        fname = self._selected_filename()
        if not fname:
            return
        ts = _format_timestamp(self._table.item(
            self._table.currentRow(), 0).text())
        ret = QMessageBox.question(
            self, "Delete Version",
            f"Permanently delete the snapshot from {ts}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        self._store.delete(fname)
        self._refresh()
