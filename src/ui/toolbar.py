from PySide6.QtWidgets import QToolBar, QWidget
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtCore import Signal


class MainToolbar(QToolBar):
    import_clicked = Signal()
    detect_cuts_clicked = Signal()
    play_pause_clicked = Signal()
    split_clicked = Signal()
    delete_clicked = Signal()
    select_to_gap_clicked = Signal()
    selection_follows_toggled = Signal(bool)
    export_video_clicked = Signal()
    export_images_clicked = Signal()
    mode_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)

        self._import_action = QAction("Import Video", self)
        self._import_action.triggered.connect(self.import_clicked.emit)
        self.addAction(self._import_action)

        self._detect_action = QAction("Detect Cuts", self)
        self._detect_action.triggered.connect(self.detect_cuts_clicked.emit)
        self.addAction(self._detect_action)

        self.addSeparator()

        self._play_action = QAction("Play", self)
        self._play_action.triggered.connect(self.play_pause_clicked.emit)
        self.addAction(self._play_action)

        self._split_action = QAction("Split", self)
        self._split_action.triggered.connect(self.split_clicked.emit)
        self.addAction(self._split_action)

        self._delete_action = QAction("Delete", self)
        self._delete_action.triggered.connect(self.delete_clicked.emit)
        self.addAction(self._delete_action)

        self.addSeparator()

        self._select_to_gap_action = QAction("Select to Gap", self)
        self._select_to_gap_action.triggered.connect(self.select_to_gap_clicked.emit)
        self.addAction(self._select_to_gap_action)

        self._selection_follows_action = QAction("Selection Follows Playhead", self)
        self._selection_follows_action.setCheckable(True)
        self._selection_follows_action.setChecked(True)
        self._selection_follows_action.toggled.connect(self.selection_follows_toggled.emit)
        self.addAction(self._selection_follows_action)

        self.addSeparator()

        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)

        self._selection_mode_action = QAction("Selection", self)
        self._selection_mode_action.setCheckable(True)
        self._selection_mode_action.setChecked(True)
        self._mode_group.addAction(self._selection_mode_action)
        self.addAction(self._selection_mode_action)

        self._cut_mode_action = QAction("Cut", self)
        self._cut_mode_action.setCheckable(True)
        self._mode_group.addAction(self._cut_mode_action)
        self.addAction(self._cut_mode_action)

        self._mode_group.triggered.connect(self._on_mode_action_triggered)

        self.addSeparator()

        self._export_video_action = QAction("Export Video", self)
        self._export_video_action.triggered.connect(self.export_video_clicked.emit)
        self.addAction(self._export_video_action)

        self._export_images_action = QAction("Export Images", self)
        self._export_images_action.triggered.connect(self.export_images_clicked.emit)
        self.addAction(self._export_images_action)

    def _on_mode_action_triggered(self, action):
        if action == self._selection_mode_action:
            self.mode_changed.emit(0)
        elif action == self._cut_mode_action:
            self.mode_changed.emit(1)

    def set_mode(self, mode: int):
        """Update toolbar toggle state without emitting mode_changed."""
        self._mode_group.blockSignals(True)
        if mode == 0:
            self._selection_mode_action.setChecked(True)
        elif mode == 1:
            self._cut_mode_action.setChecked(True)
        self._mode_group.blockSignals(False)

    def set_playing(self, playing: bool):
        self._play_action.setText("Pause" if playing else "Play")

    @property
    def selection_follows_playhead(self) -> bool:
        return self._selection_follows_action.isChecked()
