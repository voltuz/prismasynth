"""Confirmation dialog for removing a source from the Media Pool.

Three explicit choices:
  - Cancel
  - Remove source AND all its clips on the timeline
  - Remove source, keep clips on the timeline (clips become orphaned and
    render black, matching the existing missing-source behaviour)

Returns one of CANCEL / REMOVE_WITH_CLIPS / REMOVE_KEEP_CLIPS via .action.
"""

from enum import Enum

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from core.ui_scale import ui_scale


class RemoveSourceAction(Enum):
    CANCEL = "cancel"
    REMOVE_WITH_CLIPS = "remove_with_clips"
    REMOVE_KEEP_CLIPS = "remove_keep_clips"


class RemoveSourceDialog(QDialog):
    def __init__(self, source_name: str, clip_count: int,
                 source_count: int = 1, parent=None):
        super().__init__(parent)
        title = "Remove Source" if source_count == 1 else "Remove Sources"
        self.setWindowTitle(title)
        self.setMinimumWidth(ui_scale().px(440))
        self.setModal(True)
        self.action: RemoveSourceAction = RemoveSourceAction.CANCEL

        layout = QVBoxLayout(self)

        clip_word = "clip" if clip_count == 1 else "clips"
        if source_count == 1:
            msg = QLabel(
                f"<b>{source_name}</b> is referenced by <b>{clip_count}</b> "
                f"{clip_word} on the timeline.<br><br>"
                f"How would you like to proceed?"
            )
            keep_label = (
                f"Remove source — keep {clip_count} {clip_word} on timeline")
            with_label = (
                f"Remove source AND its {clip_count} {clip_word}")
        else:
            verb = "reference" if source_count != 1 else "references"
            msg = QLabel(
                f"<b>{source_count} sources</b> {verb} <b>{clip_count}</b> "
                f"{clip_word} on the timeline.<br><br>"
                f"How would you like to proceed?"
            )
            keep_label = (
                f"Remove {source_count} sources — keep {clip_count} "
                f"{clip_word} on timeline")
            with_label = (
                f"Remove {source_count} sources AND their {clip_count} "
                f"{clip_word}")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        layout.addSpacing(8)

        # Stacked buttons (one per row) — clearer than a tight horizontal row
        keep_btn = QPushButton(keep_label)
        keep_btn.setToolTip(
            "Clips will remain on the timeline but render black and have no "
            "preview, matching the missing-source state. You can relink to a "
            "new file later."
        )
        keep_btn.clicked.connect(self._on_keep)
        layout.addWidget(keep_btn)

        with_btn = QPushButton(with_label)
        with_btn.setStyleSheet("color: #e87a75;")
        with_btn.setToolTip("Permanently removes both the source(s) and every clip referencing them.")
        with_btn.clicked.connect(self._on_remove_with)
        layout.addWidget(with_btn)

        layout.addSpacing(4)

        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_row.addWidget(cancel_btn)
        layout.addLayout(cancel_row)

    def _on_keep(self):
        self.action = RemoveSourceAction.REMOVE_KEEP_CLIPS
        self.accept()

    def _on_remove_with(self):
        self.action = RemoveSourceAction.REMOVE_WITH_CLIPS
        self.accept()
