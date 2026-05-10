"""Cache Manager dialog — disk usage breakdown + per-category clear.

Surfaces every cache PrismaSynth writes to under ``%LOCALAPPDATA%/prismasynth/``
so the user can reclaim space on demand. Categories:

- Source thumbnails (``cache/source_thumbs/``)
- Proxy files (``cache/proxies/``)
- Disk thumbnails (``cache/thumbs/``)
- OmniShotCut model checkpoint (``models/OmniShotCut_ckpt.pth``)

Proxies are mmap'd while the project is loaded, so on Windows the OS won't
let us delete them. The dialog calls ``proxy_manager.close_all()`` before
clearing that category to release locks; ``ProxyManager`` re-opens proxies
on next demand.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView,
)

from utils.paths import get_cache_dir
from core.ui_scale import ui_scale

logger = logging.getLogger(__name__)


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    val = float(num_bytes) / 1024.0
    for unit in ("KB", "MB", "GB", "TB"):
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


# Per-category descriptors: (display name, subpath under prismasynth/,
# is_directory, warn_text shown in the confirm dialog).
_CATEGORIES = [
    ("Source Thumbnails", "cache/source_thumbs", True, None),
    ("Proxy Files",       "cache/proxies",       True, None),
    ("Disk Thumbnails",   "cache/thumbs",        True, None),
    ("OmniShotCut Model", "models/OmniShotCut_ckpt.pth", False,
     "OmniShotCut detection won't work until you re-run the setup "
     "(scripts/setup_omnishotcut.py or the in-app Detect Cuts setup)."),
]


class CacheManagerDialog(QDialog):
    """Modal disk-usage report with per-category clear and an Open Folder
    shortcut to ``%LOCALAPPDATA%/prismasynth/``."""

    def __init__(self, proxy_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cache Manager")
        self.setModal(True)
        _s = ui_scale()
        self.setMinimumWidth(_s.px(600))
        self.setMinimumHeight(_s.px(300))

        self._proxy_manager = proxy_manager
        # get_cache_dir() returns %LOCALAPPDATA%/prismasynth/cache/.
        # The model lives one level up under models/, so the open-folder
        # button targets the parent (%LOCALAPPDATA%/prismasynth/).
        self._app_root = get_cache_dir().parent

        layout = QVBoxLayout(self)

        intro = QLabel(
            "PrismaSynth caches thumbnails, proxies, and ML models on disk. "
            "Clearing only frees space — caches regenerate on demand.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(len(_CATEGORIES), 4, self)
        self._table.setHorizontalHeaderLabels(["Category", "Files", "Size", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self._table)

        for row, (name, _subpath, _is_dir, _warn) in enumerate(_CATEGORIES):
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(""))
            self._table.setItem(row, 2, QTableWidgetItem(""))
            btn = QPushButton("Clear")
            btn.clicked.connect(lambda _checked=False, r=row: self._clear_row(r))
            self._table.setCellWidget(row, 3, btn)

        self._total_label = QLabel("")
        bottom = QHBoxLayout()
        bottom.addWidget(self._total_label)
        bottom.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        bottom.addWidget(refresh_btn)
        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self._open_folder)
        bottom.addWidget(open_btn)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        self._refresh()

    # ------------------------------------------------------------------

    def _resolve(self, subpath: str) -> Path:
        return self._app_root / subpath

    def _walk(self, row: int) -> Tuple[int, int]:
        """Return (file_count, total_bytes) for a category."""
        _name, subpath, is_dir, _warn = _CATEGORIES[row]
        path = self._resolve(subpath)
        if not path.exists():
            return (0, 0)
        if not is_dir:
            try:
                return (1, path.stat().st_size)
            except OSError:
                return (0, 0)
        count, total = 0, 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                        count += 1
                    except OSError:
                        pass
        except OSError as e:
            logger.warning("walk %s failed: %s", path, e)
        return (count, total)

    def _refresh(self):
        grand = 0
        for row in range(len(_CATEGORIES)):
            count, total = self._walk(row)
            self._table.item(row, 1).setText(str(count))
            self._table.item(row, 2).setText(_format_size(total))
            grand += total
        self._total_label.setText(f"Total: {_format_size(grand)}")

    def _clear_row(self, row: int):
        name, subpath, is_dir, warn = _CATEGORIES[row]
        path = self._resolve(subpath)
        count, total = self._walk(row)
        if count == 0:
            QMessageBox.information(self, name, f"No files in {name}.")
            return

        body = (f"Delete {count} file(s) ({_format_size(total)}) from\n"
                f"{path}?")
        if warn:
            body += f"\n\n{warn}"
        ret = QMessageBox.question(
            self, f"Clear {name}", body,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        # Proxies are mmap'd while the project is loaded; release locks first
        # so Windows lets us unlink. ProxyManager will lazy-reopen on demand.
        if "proxies" in subpath and self._proxy_manager is not None:
            try:
                self._proxy_manager.close_all()
            except Exception as e:
                logger.warning("close_all proxies failed: %s", e)

        deleted, freed, errors = self._delete(path, is_dir)
        self._refresh()
        if errors:
            QMessageBox.warning(
                self, name,
                f"Cleared {deleted} file(s) ({_format_size(freed)}). "
                f"{errors} file(s) could not be deleted (in use).")

    def _delete(self, path: Path, is_dir: bool) -> Tuple[int, int, int]:
        """Delete every file under ``path``. Returns (deleted, freed, errors)."""
        if not path.exists():
            return (0, 0, 0)
        if not is_dir:
            try:
                size = path.stat().st_size
                path.unlink()
                return (1, size, 0)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", path, e)
                return (0, 0, 1)
        deleted, freed, errors = 0, 0, 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    deleted += 1
                    freed += sz
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", p, e)
                    errors += 1
        # Prune empty subdirectories (e.g. cache/thumbs/<source-id>/).
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        return (deleted, freed, errors)

    def _open_folder(self):
        target = self._app_root
        try:
            target.mkdir(parents=True, exist_ok=True)
            os.startfile(str(target))
        except Exception as e:
            QMessageBox.warning(
                self, "Open Folder",
                f"Could not open {target}:\n{e}")
