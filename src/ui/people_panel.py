"""People panel — manages the project's group registry.

Rows show: colour swatch, name (editable), digit (0-9 / none), clip count,
delete. Click `+ Add Group` to create a new one (auto-picks the first free
digit + a palette colour). The timeline-side keyboard shortcuts ``0``-``9``
toggle the selected clips into / out of the group bound to that digit; this
panel is also the only path to manage groups without a digit.
"""

from typing import Dict, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget,
)

from core.group import Group
from core.timeline import TimelineModel
from core.ui_scale import ui_scale
from ui.icon_loader import icon


_DIGIT_NONE = "—"


def _digit_label(d: Optional[int]) -> str:
    return _DIGIT_NONE if d is None else str(d)


def _readable_text_color(hex_str: str) -> str:
    """Return '#000' or '#fff' depending on which contrasts better with a
    given hex colour. Used for the colour-swatch button label."""
    c = QColor(hex_str)
    # Relative luminance per W3C: 0.2126 R + 0.7152 G + 0.0722 B
    lum = (0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF())
    return "#000" if lum > 0.55 else "#fff"


class _GroupRow(QWidget):
    """One row in the panel: swatch / name / digit / count / delete."""

    delete_requested = Signal(str)            # group_id
    name_changed = Signal(str, str)            # group_id, new name
    color_changed = Signal(str, str)           # group_id, new hex
    digit_changed = Signal(str, object)        # group_id, new digit (int or None)

    def __init__(self, group: Group, clip_count: int, parent=None):
        super().__init__(parent)
        self._group_id = group.id
        # _digit_combo's setCurrentIndex fires currentIndexChanged; this flag
        # lets us suppress the signal during programmatic refresh.
        self._suppress_signals = False

        s = ui_scale()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(s.px(6))

        self._swatch = QPushButton()
        self._swatch.setFixedSize(s.px(24), s.px(24))
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swatch.setToolTip("Click to change colour")
        self._swatch.clicked.connect(self._on_swatch_clicked)
        layout.addWidget(self._swatch)

        self._name_edit = QLineEdit(group.name)
        self._name_edit.setPlaceholderText("Group name")
        self._name_edit.editingFinished.connect(self._on_name_committed)
        layout.addWidget(self._name_edit, 1)

        self._digit_combo = QComboBox()
        self._digit_combo.addItem(_DIGIT_NONE, None)
        # Keyboard number-row order: 1-9 first, 0 last.
        for d in (1, 2, 3, 4, 5, 6, 7, 8, 9, 0):
            self._digit_combo.addItem(str(d), d)
        self._digit_combo.setFixedWidth(s.px(56))
        self._digit_combo.setToolTip("Keyboard digit (0-9) bound to this group")
        self._digit_combo.currentIndexChanged.connect(self._on_digit_changed)
        layout.addWidget(self._digit_combo)

        self._count_label = QLabel()
        self._count_label.setMinimumWidth(s.px(56))
        self._count_label.setStyleSheet("color: #aaa;")
        self._count_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._count_label)

        self._delete_btn = QPushButton()
        self._delete_btn.setIcon(icon("trash"))
        self._delete_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._delete_btn.setFixedSize(s.px(22), s.px(22))
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Remove group")
        self._delete_btn.setStyleSheet(
            "QPushButton { background: transparent;"
            " border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { background: rgba(232, 122, 117, 40); }")
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._group_id))
        layout.addWidget(self._delete_btn)

        self.update_from(group, clip_count)

    def refresh_scale(self):
        """Re-apply scaled sizes after a UI-scale change."""
        s = ui_scale()
        self.layout().setSpacing(s.px(6))
        self._swatch.setFixedSize(s.px(24), s.px(24))
        self._digit_combo.setFixedWidth(s.px(56))
        self._count_label.setMinimumWidth(s.px(56))
        self._delete_btn.setIconSize(QSize(s.px(14), s.px(14)))
        self._delete_btn.setFixedSize(s.px(22), s.px(22))

    def update_from(self, group: Group, clip_count: int):
        """Refresh widgets from the latest group state. Suppresses signals
        so we don't fire change events during a refresh."""
        self._suppress_signals = True
        try:
            self._swatch.setStyleSheet(
                f"QPushButton {{ background-color: {group.color};"
                f" border: 1px solid #555; border-radius: 3px;"
                f" color: {_readable_text_color(group.color)}; }}")
            if self._name_edit.text() != group.name:
                self._name_edit.setText(group.name)
            # Find the entry by stored data so the order in the combo box
            # stays decoupled from the digit value.
            idx = self._digit_combo.findData(group.digit)
            if idx >= 0 and self._digit_combo.currentIndex() != idx:
                self._digit_combo.setCurrentIndex(idx)
            self._count_label.setText(
                f"{clip_count} clip" if clip_count == 1 else f"{clip_count} clips")
        finally:
            self._suppress_signals = False

    def _on_swatch_clicked(self):
        cur = self._swatch.styleSheet()
        # Pull the current colour out of the live group via the parent's
        # query — but for simplicity we just open a dialog with the swatch's
        # actual rendered colour.
        c = QColorDialog.getColor(parent=self, title="Group Colour")
        if c.isValid():
            self.color_changed.emit(self._group_id, c.name())

    def _on_name_committed(self):
        if self._suppress_signals:
            return
        self.name_changed.emit(self._group_id, self._name_edit.text().strip())

    def _on_digit_changed(self, idx: int):
        if self._suppress_signals:
            return
        data = self._digit_combo.itemData(idx)
        self.digit_changed.emit(self._group_id, data)


class PeoplePanel(QWidget):
    """The People (groups) panel — lives in the right column behind a tab
    switcher next to the Clip Info panel."""

    # Reserved for a future "filter timeline by group" feature.
    group_clicked = Signal(str)

    def __init__(self, timeline: TimelineModel, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._rows: Dict[str, _GroupRow] = {}

        s = ui_scale()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(s.px(8), s.px(8), s.px(8), s.px(8))
        outer.setSpacing(s.px(6))
        self._outer_layout = outer

        # Header
        hdr = QHBoxLayout()
        title = QLabel("People")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ddd;")
        hdr.addWidget(title)
        hdr.addStretch()
        add_btn = QPushButton("+ Add Group")
        add_btn.clicked.connect(self._on_add_group)
        hdr.addWidget(add_btn)
        outer.addLayout(hdr)

        # Scrollable rows area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(s.px(4))
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_host)
        outer.addWidget(self._scroll, 1)

        # Empty-state hint
        self._empty_label = QLabel(
            "No groups yet. Click \"+ Add Group\" or press a digit (0-9) "
            "with one or more clips selected to create one.")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: #888; font-size: 11px;")
        outer.addWidget(self._empty_label)

        # Refresh hooks
        self._timeline.groups_changed.connect(self._refresh)
        self._timeline.clips_changed.connect(self._refresh_counts)
        ui_scale().changed.connect(self._on_ui_scale_changed)

        self._refresh()

    def _on_ui_scale_changed(self):
        s = ui_scale()
        self._outer_layout.setContentsMargins(
            s.px(8), s.px(8), s.px(8), s.px(8))
        self._outer_layout.setSpacing(s.px(6))
        self._rows_layout.setSpacing(s.px(4))
        for row in self._rows.values():
            row.refresh_scale()

    # ------------------------------------------------------------------

    def _clip_count_for(self, group_id: str) -> int:
        return sum(1 for c in self._timeline.clips
                   if not c.is_gap and group_id in c.group_ids)

    def _refresh(self):
        groups = self._timeline.groups
        # Remove rows for deleted groups
        for gid in list(self._rows.keys()):
            if gid not in groups:
                row = self._rows.pop(gid)
                self._rows_layout.removeWidget(row)
                row.deleteLater()
        # Add or update rows in registry order
        existing_set = set(self._rows.keys())
        for gid, g in groups.items():
            count = self._clip_count_for(gid)
            if gid in self._rows:
                self._rows[gid].update_from(g, count)
            else:
                row = _GroupRow(g, count, parent=self._rows_host)
                row.name_changed.connect(self._on_row_name_changed)
                row.color_changed.connect(self._on_row_color_changed)
                row.digit_changed.connect(self._on_row_digit_changed)
                row.delete_requested.connect(self._on_row_delete)
                # Insert before the trailing stretch.
                idx = self._rows_layout.count() - 1
                self._rows_layout.insertWidget(idx, row)
                self._rows[gid] = row
        self._empty_label.setVisible(len(groups) == 0)

    def _refresh_counts(self):
        # Cheaper variant — only update clip counts on each row. ``_rows``
        # can briefly hold stale gids when ``TimelineModel.clear()`` /
        # ``_restore`` empties _groups: clips_changed fires BEFORE
        # groups_changed, so this slot runs first while the row cache still
        # mirrors the previous project. Skip stale rows here — the
        # groups_changed emit a moment later runs _refresh and reconciles.
        groups = self._timeline.groups
        for gid, row in self._rows.items():
            if gid in groups:
                row.update_from(groups[gid], self._clip_count_for(gid))

    # --- Row callbacks ---------------------------------------------------

    def _on_add_group(self):
        # Auto-pick the first free digit. Keyboard-row order means 1 is
        # tried first, then 2…9, then 0. None if all 10 are taken.
        used = {g.digit for g in self._timeline.groups.values()
                if g.digit is not None}
        digit = next(
            (d for d in (1, 2, 3, 4, 5, 6, 7, 8, 9, 0) if d not in used),
            None)
        # Find the next sequentially-free default name "Group N".
        n = 1
        while any(g.name == f"Group {n}"
                  for g in self._timeline.groups.values()):
            n += 1
        self._timeline.add_group(name=f"Group {n}", digit=digit)

    def _on_row_name_changed(self, group_id: str, new_name: str):
        if not new_name:
            # Don't allow blank names — refresh to revert the field.
            self._refresh()
            return
        self._timeline.update_group(group_id, name=new_name)

    def _on_row_color_changed(self, group_id: str, new_hex: str):
        self._timeline.update_group(group_id, color=new_hex)

    def _on_row_digit_changed(self, group_id: str, new_digit):
        # ``new_digit`` is None for "—", or an int 0-9.
        conflict = self._timeline.update_group(group_id, digit=new_digit)
        if conflict is not None:
            QMessageBox.warning(
                self, "Digit In Use",
                f"Digit {new_digit} is already used by '{conflict}'.\n"
                f"Clear or change that one first.")
            # Revert the combo to the actual current value.
            self._refresh()

    def _on_row_delete(self, group_id: str):
        g = self._timeline.groups.get(group_id)
        if g is None:
            return
        n = self._clip_count_for(group_id)
        msg = f"Remove group '{g.name}'?"
        if n > 0:
            msg += f"\n\nIt currently tags {n} clip{'s' if n != 1 else ''}."
        ret = QMessageBox.question(
            self, "Remove Group", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if ret == QMessageBox.StandardButton.Yes:
            self._timeline.remove_group(group_id)
