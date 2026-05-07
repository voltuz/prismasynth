"""Group filter widget — embedded in every export dialog so the user
can restrict the export to clips belonging to a chosen subset of People
groups (and/or untagged clips).

Filter encoding (matches ``core.group.clip_matches_filter``):

- No checkbox ticked → ``current_filter()`` returns ``None`` (no filter,
  export everything — backward-compatible default).
- Any checkbox ticked → filter is active; clips export only when they
  match at least one ticked entry. The "(Untagged)" row covers clips
  that aren't in any group.

The widget hides itself when the timeline has zero groups, since the
only meaningful row would be "(Untagged)" and unchecked = "no filter"
anyway — there's nothing useful to interact with.
"""

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from core.group import Group
from core.timeline import TimelineModel


_UNTAGGED_KEY = "__untagged__"


class _Swatch(QLabel):
    """Small filled colour square used as the per-group row marker."""
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.setStyleSheet(
            f"background-color: {color};"
            f" border: 1px solid #555; border-radius: 2px;")


class GroupFilterWidget(QWidget):
    """Reusable filter UI for the export dialogs.

    Emits ``selection_changed`` whenever any checkbox is toggled so the
    parent dialog can refresh its info label and Export-button state.
    """

    selection_changed = Signal()

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        # key -> QCheckBox. Key = group_id for groups, _UNTAGGED_KEY for
        # the special "(Untagged)" row.
        self._checks: Dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QLabel(
            "Groups to export — uncheck = no filter, "
            "check any to limit to those:")
        header.setStyleSheet("color: #aaa;")
        header.setWordWrap(True)
        outer.addWidget(header)

        # Scroll area in case there are many groups.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setMaximumHeight(140)
        host = QWidget()
        self._rows_layout = QVBoxLayout(host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll)

        self._rebuild_rows()

        # If the registry changes mid-dialog (unlikely but possible if the
        # user opens the People tab at the same time), refresh the rows.
        self._timeline.groups_changed.connect(self._rebuild_rows)

    # ------------------------------------------------------------------

    def current_filter(self) -> Optional[dict]:
        """Snapshot the current selection. Returns ``None`` when nothing is
        ticked (i.e. no filter is active)."""
        any_checked = any(cb.isChecked() for cb in self._checks.values())
        if not any_checked:
            return None
        return {
            "group_ids": [
                k for k, cb in self._checks.items()
                if k != _UNTAGGED_KEY and cb.isChecked()
            ],
            "include_untagged": (
                _UNTAGGED_KEY in self._checks
                and self._checks[_UNTAGGED_KEY].isChecked()
            ),
        }

    # ------------------------------------------------------------------

    def _rebuild_rows(self):
        # Remove existing rows + checkboxes
        while self._rows_layout.count() > 1:
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checks.clear()

        groups = self._timeline.groups
        if not groups:
            # Hide the whole widget when there are no groups — the only
            # remaining checkbox would be "(Untagged)" and unchecked =
            # "no filter" anyway, so nothing useful.
            self.setVisible(False)
            return
        self.setVisible(True)

        # "(Untagged)" first, then groups in registry order.
        self._add_row(_UNTAGGED_KEY, "(Untagged)", color=None,
                      digit=None, italic=True)
        for gid, g in groups.items():
            self._add_row(gid, g.name, color=g.color, digit=g.digit,
                          italic=False)

    def _add_row(self, key: str, name: str, color: Optional[str],
                 digit: Optional[int], italic: bool):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        cb = QCheckBox()
        cb.toggled.connect(self.selection_changed)
        row_layout.addWidget(cb)

        if color is not None:
            row_layout.addWidget(_Swatch(color))
        else:
            # Reserve the same horizontal space the swatch would take so
            # the (Untagged) row's text starts at the same x as group rows.
            spacer = QLabel("")
            spacer.setFixedSize(12, 12)
            row_layout.addWidget(spacer)

        label = QLabel(name)
        if italic:
            f = label.font()
            f.setItalic(True)
            label.setFont(f)
            label.setStyleSheet("color: #aaa;")
        row_layout.addWidget(label, 1)

        if digit is not None:
            digit_label = QLabel(f"[{digit}]")
            digit_label.setStyleSheet("color: #888;")
            row_layout.addWidget(digit_label)

        # Insert before the trailing stretch.
        idx = self._rows_layout.count() - 1
        self._rows_layout.insertWidget(idx, row)
        self._checks[key] = cb
