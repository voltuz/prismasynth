"""Keyboard Shortcuts dialog.

Lists every entry in ``core.shortcuts.SHORTCUTS`` grouped by category and
lets the user rebind, clear, or reset to default. Reassignment is via a
small capture modal: the next non-modifier key press becomes the new
sequence. Conflicts are rejected with a name-the-conflict message — by
design, every shortcut must have a unique key.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from core.shortcuts import ShortcutManager


class _CaptureDialog(QDialog):
    """Modal that records the next non-modifier key combo the user presses."""

    def __init__(self, action_name: str, current: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Press Shortcut")
        self.setModal(True)
        self.setMinimumWidth(380)
        self._captured: str = ""

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Rebinding: <b>{action_name}</b>"))
        layout.addSpacing(4)
        cur_label = QLabel(f"Current: {current or '(none)'}")
        cur_label.setStyleSheet("color: #aaa;")
        layout.addWidget(cur_label)
        layout.addSpacing(8)
        prompt = QLabel("Press the new shortcut.\nEsc to cancel.")
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        # We need keyboard focus to receive the keys.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def captured(self) -> str:
        return self._captured

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.reject()
            return
        # Modifier-only presses are noise — wait for a real key.
        if key in (Qt.Key.Key_Shift, Qt.Key.Key_Control,
                   Qt.Key.Key_Alt, Qt.Key.Key_Meta,
                   Qt.Key.Key_AltGr, Qt.Key.Key_CapsLock):
            return
        combo = int(event.modifiers().value) | int(key)
        seq = QKeySequence(combo).toString()
        if seq:
            self._captured = seq
            self.accept()


class KeyboardShortcutsDialog(QDialog):
    """Tree view of every customizable shortcut, grouped by category."""

    def __init__(self, manager: ShortcutManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self.setModal(True)
        self._manager = manager

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Double-click a row (or use Edit) to rebind. Each shortcut must "
            "be unique — if you assign a key already in use, the dialog will "
            "tell you which action holds it.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Action", "Shortcut"])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        h = self._tree.header()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._tree)

        # Bottom buttons
        btn_row = QHBoxLayout()
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(edit_btn)
        reset_one_btn = QPushButton("Reset to default")
        reset_one_btn.clicked.connect(self._on_reset_one)
        btn_row.addWidget(reset_one_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        reset_all_btn = QPushButton("Reset All to Defaults")
        reset_all_btn.clicked.connect(self._on_reset_all)
        btn_row.addWidget(reset_all_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._tree.itemDoubleClicked.connect(
            lambda item, _col: self._edit_id(
                item.data(0, Qt.ItemDataRole.UserRole)))

        self._populate()

    # ------------------------------------------------------------------

    def _populate(self):
        self._tree.clear()
        # Group by category; rely on registry order for display order within
        # each category.
        cats: dict = {}
        order: list = []
        for d, current in self._manager.get_all():
            if d.category not in cats:
                cats[d.category] = []
                order.append(d.category)
            cats[d.category].append((d, current))
        for cat in order:
            cat_item = QTreeWidgetItem([cat, ""])
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._tree.addTopLevelItem(cat_item)
            for d, current in cats[cat]:
                row = QTreeWidgetItem([d.name, current or "(none)"])
                row.setData(0, Qt.ItemDataRole.UserRole, d.id)
                cat_item.addChild(row)
            cat_item.setExpanded(True)

    def _selected_id(self) -> Optional[str]:
        items = self._tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def _on_edit(self):
        sid = self._selected_id()
        if sid:
            self._edit_id(sid)

    def _edit_id(self, sid: Optional[str]):
        if not sid:
            return
        action_name = next(
            (d.name for d, _ in self._manager.get_all() if d.id == sid), sid)
        current = self._manager.get(sid)
        cap = _CaptureDialog(action_name, current, parent=self)
        if cap.exec() != QDialog.DialogCode.Accepted:
            return
        new = cap.captured
        if not new or new == current:
            return
        conflict = self._manager.set_key(sid, new)
        if conflict:
            QMessageBox.warning(
                self, "Shortcut In Use",
                f"'{new}' is already used by '{conflict}'.\n\n"
                f"Pick a different key, or clear the other shortcut first.")
            # Re-open so the user can try again immediately.
            self._edit_id(sid)
            return
        self._populate()

    def _on_reset_one(self):
        sid = self._selected_id()
        if not sid:
            return
        conflict = self._manager.reset_one(sid)
        if conflict:
            QMessageBox.warning(
                self, "Shortcut In Use",
                f"Can't reset — the default key for this action is "
                f"already used by '{conflict}'. "
                f"Clear or rebind that one first.")
            return
        self._populate()

    def _on_clear(self):
        sid = self._selected_id()
        if not sid:
            return
        self._manager.set_key(sid, "")
        self._populate()

    def _on_reset_all(self):
        ret = QMessageBox.question(
            self, "Reset All",
            "Reset every shortcut to its default? "
            "This will overwrite any custom assignments.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self._manager.reset_all()
        self._populate()
