import logging
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QWidget, QAbstractItemView,
)

from core.video_source import VideoSource
from utils.ffprobe import probe_video, VideoInfo

logger = logging.getLogger(__name__)

_FPS_TOLERANCE = 0.02
_FRAME_TOLERANCE = 0.01  # ±1%


_STATUS_LINKED = "Linked"
_STATUS_MISSING = "Missing"
_LINKED_COLOR = QColor("#6cba7e")
_MISSING_COLOR = QColor("#e8a735")


class RelinkDialog(QDialog):
    """Modal dialog for managing source file paths.

    Originally built for missing-source recovery on project load; now also
    invoked from the Media Pool to repoint an already-existing source at a
    different file. The status column reflects the current state per source —
    "Linked" (green) when the file exists, "Missing" (orange) when not."""

    def __init__(self, missing: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Relink Sources")
        self.setModal(True)
        self.setMinimumWidth(720)
        self.setMinimumHeight(380)

        self._missing: dict[str, VideoSource] = dict(missing)
        self._resolved: dict[str, Optional[str]] = {}
        self._probe_cache: dict[str, VideoInfo] = {}
        self._row_by_id: dict[str, int] = {}
        self._last_browse_dir: str = ""

        # Pre-compute which sources are missing so we can phrase the intro
        # appropriately and gate the OK button correctly.
        self._missing_count = sum(
            1 for s in self._missing.values() if not os.path.exists(s.file_path)
        )
        total = len(self._missing)

        layout = QVBoxLayout(self)

        if self._missing_count == total:
            intro_text = (f"{total} source file(s) could not be found. "
                          "Browse to relink them, or skip to keep broken references.")
        elif self._missing_count == 0:
            intro_text = ("Pick a different file for any source you want to "
                          "repoint, or close to keep the existing links.")
        else:
            intro_text = (f"{self._missing_count} of {total} source file(s) are missing. "
                          "Browse to relink missing ones, or repoint existing ones.")
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(len(self._missing), 4, self)
        self._table.setHorizontalHeaderLabels(["File", "Status", "", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self._table)

        for row, (sid, src) in enumerate(self._missing.items()):
            self._row_by_id[sid] = row
            basename = os.path.basename(src.file_path) or src.file_path
            file_item = QTableWidgetItem(basename)
            file_item.setToolTip(src.file_path)
            self._table.setItem(row, 0, file_item)

            # Reflect the actual on-disk state per source.
            if os.path.exists(src.file_path):
                status_text, status_color = _STATUS_LINKED, _LINKED_COLOR
            else:
                status_text, status_color = _STATUS_MISSING, _MISSING_COLOR
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(status_color))
            status_item.setToolTip(src.file_path)
            self._table.setItem(row, 1, status_item)

            browse_btn = QPushButton("Browse…")
            browse_btn.clicked.connect(lambda _=False, s=sid: self._browse_row(s))
            self._table.setCellWidget(row, 2, browse_btn)

            skip_cell = QWidget()
            skip_layout = QHBoxLayout(skip_cell)
            skip_layout.setContentsMargins(8, 0, 8, 0)
            skip_cb = QCheckBox("Skip")
            skip_cb.toggled.connect(lambda checked, s=sid: self._toggle_skip(s, checked))
            skip_layout.addWidget(skip_cb)
            self._table.setCellWidget(row, 3, skip_cell)

        bottom = QHBoxLayout()
        self._rebase_btn = QPushButton("Browse for All from Folder…")
        self._rebase_btn.setToolTip(
            "Pick a folder. PrismaSynth will auto-find every missing source "
            "by filename in that folder."
        )
        self._rebase_btn.clicked.connect(self._browse_for_all)
        bottom.addWidget(self._rebase_btn)
        bottom.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(self._cancel_btn)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self.accept)
        bottom.addWidget(self._ok_btn)
        layout.addLayout(bottom)
        # OK is enabled when every source is either explicitly handled
        # (relinked or skipped) OR already linked to an existing file.
        self._update_ok_enabled()

    def resolved_paths(self) -> dict:
        return dict(self._resolved)

    def probe_cache(self) -> dict:
        return dict(self._probe_cache)

    def _browse_row(self, source_id: str):
        src = self._missing[source_id]
        target_basename = os.path.basename(src.file_path)
        start_dir = self._last_browse_dir or (
            os.path.dirname(src.file_path) if src.file_path else ""
        )
        folder = QFileDialog.getExistingDirectory(
            self, f"Select folder containing {target_basename}", start_dir,
        )
        if not folder:
            return
        self._last_browse_dir = folder

        try:
            entries = os.listdir(folder)
        except OSError as e:
            QMessageBox.critical(self, "Folder Error", f"Could not read folder:\n{e}")
            return
        folder_index = {e.lower(): e for e in entries}
        target_lower = target_basename.lower()
        if target_lower not in folder_index:
            QMessageBox.warning(
                self, "Not Found",
                f"Could not find '{target_basename}' in:\n{folder}",
            )
            return

        candidate = os.path.join(folder, folder_index[target_lower])
        if self._validate_and_set(source_id, candidate):
            self._folder_rebase(folder, exclude=source_id)
            self._update_ok_enabled()

    def _browse_for_all(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing missing source(s)",
            self._last_browse_dir,
        )
        if not folder:
            return
        self._last_browse_dir = folder
        self._folder_rebase(folder, exclude="")
        self._update_ok_enabled()

    def _toggle_skip(self, source_id: str, checked: bool):
        if checked:
            self._resolved[source_id] = None
            self._probe_cache.pop(source_id, None)
            self._set_status(source_id, "Skipped")
        else:
            self._resolved.pop(source_id, None)
            src = self._missing[source_id]
            if os.path.exists(src.file_path):
                self._set_status(source_id, _STATUS_LINKED,
                                 tooltip=src.file_path, color=_LINKED_COLOR)
            else:
                self._set_status(source_id, _STATUS_MISSING,
                                 tooltip=src.file_path, color=_MISSING_COLOR)
        self._update_ok_enabled()

    def _folder_rebase(self, folder: str, exclude: str):
        try:
            entries = os.listdir(folder)
        except OSError:
            return
        folder_index = {e.lower(): e for e in entries}
        for sid, src in self._missing.items():
            if sid == exclude:
                continue
            if sid in self._resolved:
                continue
            target = os.path.basename(src.file_path).lower()
            if not target:
                continue
            if target in folder_index:
                candidate = os.path.join(folder, folder_index[target])
                info = self._probe_silent(candidate)
                if info is None:
                    continue
                if not self._metadata_matches(self._missing[sid], info, strict=True):
                    continue
                self._resolved[sid] = candidate
                self._probe_cache[sid] = info
                self._set_status(sid, f"Auto-found at {candidate}", tooltip=candidate)

    def _validate_and_set(self, source_id: str, new_path: str) -> bool:
        info = probe_video(new_path)
        if info is None:
            QMessageBox.critical(
                self, "Read Error",
                f"Could not read video file:\n{os.path.basename(new_path)}\n\n"
                "Is FFmpeg installed and on PATH?",
            )
            return False
        orig = self._missing[source_id]
        if not self._metadata_matches(orig, info, strict=False):
            msg = (
                "Replacement file metadata differs from the original.\n\n"
                f"Original:    {orig.width}x{orig.height}  "
                f"{orig.fps:.3f}fps  {orig.total_frames} frames\n"
                f"Replacement: {info.width}x{info.height}  "
                f"{info.fps:.3f}fps  {info.total_frames} frames\n\n"
                "Use this file anyway?"
            )
            reply = QMessageBox.warning(
                self, "Metadata Mismatch", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
        self._resolved[source_id] = new_path
        self._probe_cache[source_id] = info
        self._set_status(source_id, f"Relinked: {new_path}", tooltip=new_path)
        return True

    @staticmethod
    def _metadata_matches(orig: VideoSource, info: VideoInfo, strict: bool) -> bool:
        if info.width != orig.width or info.height != orig.height:
            return False
        if abs(info.fps - orig.fps) > _FPS_TOLERANCE:
            return False
        if orig.total_frames > 0:
            tolerance_frames = max(1, int(orig.total_frames * _FRAME_TOLERANCE))
            if abs(info.total_frames - orig.total_frames) > tolerance_frames:
                return False
        elif strict and info.total_frames <= 0:
            return False
        return True

    @staticmethod
    def _probe_silent(path: str) -> Optional[VideoInfo]:
        try:
            return probe_video(path)
        except Exception:
            logger.exception("probe_video failed for %s", path)
            return None

    def _set_status(self, source_id: str, text: str,
                    tooltip: Optional[str] = None,
                    color: Optional[QColor] = None):
        row = self._row_by_id[source_id]
        item = self._table.item(row, 1)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, 1, item)
        item.setText(text)
        item.setToolTip(tooltip or text)
        if color is not None:
            item.setForeground(QBrush(color))

    def _update_ok_enabled(self):
        # A source is "handled" if explicitly resolved (relinked or skipped)
        # OR if its current file path exists on disk (the user can close
        # without changing it).
        all_handled = True
        for sid, src in self._missing.items():
            if sid in self._resolved:
                continue
            if os.path.exists(src.file_path):
                continue
            all_handled = False
            break
        self._ok_btn.setEnabled(all_handled)
