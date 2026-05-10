import json
import logging
import os
import time
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStatusBar, QMessageBox, QFileDialog, QProgressBar,
    QApplication, QDialog, QTabWidget, QInputDialog,
)
from PySide6.QtCore import Qt, QTimer, QEvent, QRunnable, QThreadPool, QSettings, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from version import __version__
from core.ui_scale import ui_scale
from core.timeline import TimelineModel
from core.clip import Clip
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool
from core.xml_exporter import export_fcpxml
from core.otio_exporter import export_otio
from core.exporter import Exporter
from core.thumbnail_cache import ThumbnailCache
from core.proxy_cache import ProxyManager
from core.project import save_project, load_project
from core.source_thumbnail import extract_thumbnail
from utils.paths import get_config_dir
from ui.icon_loader import icon
from ui.preview_widget import PreviewWidget
from ui.timeline_widget import TimelineWidget, EditMode
from ui.toolbar import MainToolbar
from ui.clip_info_panel import ClipInfoPanel
from ui.people_panel import PeoplePanel
from ui.media_panel import MediaPanel
from ui.source_info_dialog import SourceInfoDialog
from ui.remove_source_dialog import RemoveSourceDialog, RemoveSourceAction
from ui.import_dialog import ImportDialog
from ui.detect_dialog import DetectDialog
from ui.export_dialog import ExportDialog
from ui.relink_dialog import RelinkDialog


class _SourceThumbWorker(QRunnable):
    """Background worker that extracts one source's thumbnail and emits a
    signal when done. Runs on MainWindow's QThreadPool so the import flow
    stays non-blocking."""

    def __init__(self, source: VideoSource, signal):
        super().__init__()
        self._source = source
        self._signal = signal
        self.setAutoDelete(True)

    def run(self):
        try:
            extract_thumbnail(self._source)
        except Exception:
            logger.exception("source thumbnail extraction failed")
        self._signal.emit(self._source.id)

logger = logging.getLogger(__name__)

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #ddd;
}
QToolBar {
    background-color: #333;
    border: none;
    spacing: 4px;
    padding: 2px;
}
QToolBar QToolButton {
    background-color: #444;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px 10px;
    margin: 1px;
}
QToolBar QToolButton:hover {
    background-color: #555;
}
QToolBar QToolButton:pressed {
    background-color: #666;
}
QToolBar QToolButton:checked {
    background-color: #5577aa;
    border-color: #6688bb;
}
QStatusBar {
    background-color: #333;
    color: #aaa;
}
QMenuBar {
    background-color: #333;
    color: #ddd;
}
QMenuBar::item {
    background-color: transparent;
    padding: 4px 10px;
}
QMenuBar::item:selected {
    background-color: #555;
}
QMenuBar::item:pressed {
    background-color: #5577aa;
}
QMenu {
    background-color: #333;
    color: #ddd;
    border: 1px solid #555;
    padding: 4px 0;
}
QMenu::item {
    padding: 5px 28px 5px 32px;
}
QMenu::icon {
    padding-left: 10px;
}
QMenu::item:selected {
    background-color: #5577aa;
    color: #fff;
}
QMenu::separator {
    height: 1px;
    background: #555;
    margin: 4px 8px;
}
QGroupBox {
    border: 1px solid #555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    color: #ccc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QScrollBar:horizontal {
    background: #333;
    height: 14px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #666;
    min-width: 30px;
    border-radius: 3px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar:vertical {
    background: #333;
    width: 14px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #666;
    min-height: 30px;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSplitter::handle {
    background-color: #444;
    height: 4px;
}
QProgressBar {
    border: 1px solid #555;
    border-radius: 3px;
    text-align: center;
    background-color: #333;
    color: #ddd;
}
QProgressBar::chunk {
    background-color: #5577aa;
}
QDialog {
    background-color: #2b2b2b;
    color: #ddd;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3a3a3a;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px;
}
QPushButton {
    background-color: #444;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 5px 15px;
}
QPushButton:hover {
    background-color: #555;
}
QTabWidget::pane {
    border: 1px solid #555;
    background-color: #2b2b2b;
}
QTabBar::tab {
    background-color: #3a3a3a;
    color: #ccc;
    border: 1px solid #555;
    padding: 6px 16px;
}
QTabBar::tab:selected {
    background-color: #4a4a4a;
    color: #fff;
}
"""


class MainWindow(QMainWindow):
    # Emitted by background source-thumbnail workers when an extract finishes.
    _thumb_extracted = Signal(str)  # source_id

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PrismaSynth v{__version__}")
        _s = ui_scale()
        self.setMinimumSize(_s.px(1024), _s.px(600))
        self.resize(_s.px(1400), _s.px(800))
        self.setAcceptDrops(True)

        # Core state
        self._timeline = TimelineModel()
        self._sources: Dict[str, VideoSource] = {}
        self._reader_pool = VideoReaderPool(use_gpu=True)
        self._proxy_manager = ProxyManager()
        self._exporter: Optional[Exporter] = None
        self._detect_partials: dict = {}
        self._last_clicked_clip_id: Optional[str] = None
        self._thumbnail_cache: Optional[ThumbnailCache] = None
        self._selection_follows_playhead = True
        self._playback_updating = False

        # Project state
        self._project_path: Optional[str] = None
        self._dirty = False
        self._recent_files: list = []
        self._load_recent_files()

        # Autosave timer (every 60 seconds when dirty)
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(60000)
        self._autosave_timer.timeout.connect(self._autosave)

        # Thumbnail resume timer — resumes thumbnail generation after scrubbing stops
        self._thumb_resume_timer = QTimer()
        self._thumb_resume_timer.setSingleShot(True)
        self._thumb_resume_timer.setInterval(500)
        self._thumb_resume_timer.timeout.connect(self._resume_thumbnails)
        # Debounce viewport changes (scroll) — reprioritize after 300ms idle
        self._viewport_timer = QTimer()
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(300)
        self._viewport_timer.timeout.connect(self._do_reprioritize)

        # Playback — mpv plays natively, timer syncs the timeline playhead
        self._playback_timer = QTimer()
        self._playback_timer.setInterval(16)  # ~60Hz playhead sync
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_source = None
        self._playback_clip = None
        self._playback_clip_timeline_start = 0
        self._gap_start_time = 0.0      # monotonic time when gap playback started
        self._gap_start_frame = 0       # timeline frame at gap start

        # Apply dark theme
        self.setStyleSheet(DARK_STYLE)

        # Customizable-shortcut registry. Created before menus so QActions
        # built in _setup_menus pick up any user overrides.
        from core.shortcuts import ShortcutManager
        self._shortcut_mgr = ShortcutManager()

        # Toolbar
        self._toolbar = MainToolbar(self)
        self.addToolBar(self._toolbar)
        self._toolbar.import_clicked.connect(self._on_import)
        self._toolbar.detect_cuts_clicked.connect(self._on_detect_cuts)
        self._toolbar.play_pause_clicked.connect(self._toggle_play)
        self._toolbar.split_clicked.connect(self._on_split)
        self._toolbar.delete_clicked.connect(self._on_delete)
        self._toolbar.select_to_gap_clicked.connect(self._on_select_to_gap)
        self._toolbar.selection_follows_toggled.connect(self._on_selection_follows_toggled)
        self._toolbar.export_clicked.connect(self._on_export_video)
        self._toolbar.mode_changed.connect(self._on_mode_changed)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Thin separator below the toolbar — matches the colour and height of
        # the QSplitter handles so the toolbar feels visually divorced from
        # the panels in the same way the timeline / panels divider does.
        top_separator = QWidget()
        top_separator.setFixedHeight(ui_scale().px(4))
        top_separator.setStyleSheet("background-color: #444;")
        main_layout.addWidget(top_separator)
        self._top_separator = top_separator

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Top area: media panel | preview | clip info (3-column layout)
        # Horizontal splitter so the side panels are user-resizable.
        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setHandleWidth(ui_scale().px(4))

        self._media_panel = MediaPanel()
        self._top_splitter.addWidget(self._media_panel)

        self._preview = PreviewWidget()
        self._top_splitter.addWidget(self._preview)

        # Right column: Clip Info + People, switchable via vertical
        # text-tabs on the left edge of the column. Tab strip width is
        # tightened via stylesheet so it doesn't eat horizontal space.
        self._clip_info = ClipInfoPanel()
        self._people_panel = PeoplePanel(self._timeline)
        self._right_tabs = QTabWidget()
        self._right_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self._right_tabs.addTab(self._clip_info, "Clip Info")
        self._right_tabs.addTab(self._people_panel, "People")
        # Restore last-used right-tab from QSettings (project file, when a
        # project loads, will override this via _load_from).
        _saved_tab = QSettings().value("main_window/right_tab_index")
        if _saved_tab is not None:
            try:
                _idx = int(_saved_tab)
                if 0 <= _idx < self._right_tabs.count():
                    self._right_tabs.setCurrentIndex(_idx)
            except (TypeError, ValueError):
                pass
        self._right_tabs.currentChanged.connect(self._on_right_tab_changed)
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab {"
            " padding: 6px 2px;"
            " min-height: 60px;"
            " background-color: #2b2b2b;"
            " color: #aaa;"
            " border: 1px solid #444;"
            "}"
            "QTabBar::tab:selected {"
            " background-color: #3a3a3a;"
            " color: #fff;"
            "}"
            "QTabBar::tab:hover { background-color: #333; }"
        )
        self._top_splitter.addWidget(self._right_tabs)

        # Centre column takes the remainder; side columns stay where the user puts them.
        self._top_splitter.setStretchFactor(0, 0)
        self._top_splitter.setStretchFactor(1, 1)
        self._top_splitter.setStretchFactor(2, 0)

        # Restore saved sizes (per-user, across launches) or fall back to defaults.
        # Saved sizes are absolute pixels — left untouched on UI-scale change so
        # the user's deliberate splitter drag isn't overwritten. Only the first-
        # run default scales to the current UI scale.
        _settings = QSettings()
        _saved_sizes = _settings.value("main_window/top_splitter_sizes")
        _s = ui_scale()
        _default_sizes = [_s.px(250), _s.px(800), _s.px(250)]
        if _saved_sizes:
            try:
                self._top_splitter.setSizes([int(s) for s in _saved_sizes])
            except (TypeError, ValueError):
                self._top_splitter.setSizes(_default_sizes)
        else:
            self._top_splitter.setSizes(_default_sizes)
        self._top_splitter.splitterMoved.connect(self._on_top_splitter_moved)

        splitter.addWidget(self._top_splitter)

        # Bottom area: timeline (with left/right margins)
        self._timeline_widget = TimelineWidget(self._timeline)
        # Share the source dict so drag-from-Media-Pool previews can render
        # each source's thumbnail in the drop footprint. The dict is mutated
        # in place by import/load, so the reference stays valid.
        self._timeline_widget.set_sources_ref(self._sources)
        timeline_container = QWidget()
        tl_layout = QVBoxLayout(timeline_container)
        _m = ui_scale().px(16)
        tl_layout.setContentsMargins(_m, 0, _m, 0)
        tl_layout.addWidget(self._timeline_widget)
        splitter.addWidget(timeline_container)
        self._timeline_container_layout = tl_layout

        # Restore vertical (panels|timeline) split from QSettings, falling
        # back to the UI-scaled default. Project file overrides on load.
        _default_vert = [ui_scale().px(400), ui_scale().px(200)]
        _saved_vert = _settings.value("main_window/vertical_splitter_sizes")
        if _saved_vert:
            try:
                splitter.setSizes([int(s) for s in _saved_vert])
            except (TypeError, ValueError):
                splitter.setSizes(_default_vert)
        else:
            splitter.setSizes(_default_vert)
        self._main_splitter = splitter
        splitter.splitterMoved.connect(self._on_main_splitter_moved)

        # Live UI-scale: re-apply sizes that the user can't manually drag
        # (separator, splitter handles, layout margins). Splitter sizes and
        # _clip_height are deliberately left absolute.
        ui_scale().changed.connect(self._on_ui_scale_changed)

        # Timeline signals
        self._timeline_widget.playhead_changed.connect(self._on_playhead_changed)
        self._timeline_widget.scrub_started.connect(self._preview.scrub_start)
        self._timeline_widget.scrub_ended.connect(self._preview.scrub_end)
        # Middle-mouse pan: pause the thumbnail coordinator while the user
        # is dragging the timeline so the rapid viewport reprioritization
        # doesn't thrash the prioritization queue. Resume on release.
        self._timeline_widget.pan_started.connect(self._on_timeline_pan_started)
        self._timeline_widget.pan_ended.connect(self._on_timeline_pan_ended)
        self._timeline_widget.clip_clicked.connect(self._on_clip_clicked)
        self._timeline_widget.preview_frame_requested.connect(self._on_preview_frame_requested)
        self._timeline_widget.cut_requested.connect(self._on_cut_at_frame)
        self._timeline_widget.thumbnails_toggled.connect(self._on_thumbnails_toggled)
        self._timeline_widget.hq_thumbnails_toggled.connect(self._on_hq_thumbnails_toggled)
        self._timeline_widget.cache_thumbnails_clicked.connect(self._on_cache_thumbnails_clicked)
        self._timeline_widget.strip.scroll_changed.connect(self._on_viewport_changed)
        self._timeline_widget.sources_dropped.connect(self._on_timeline_sources_dropped)
        self._timeline_widget.files_dropped.connect(self._on_timeline_files_dropped)
        self._timeline.selection_changed.connect(self._on_selection_changed)

        # Media Panel signals
        self._media_panel.source_double_clicked.connect(self._on_source_double_clicked)
        self._media_panel.relink_requested.connect(self._on_source_relink_requested)
        self._media_panel.remove_requested.connect(self._on_source_remove_requested)
        self._media_panel.files_dropped.connect(self._import_files)

        # Source-thumbnail extraction worker pool
        self._thumb_pool = QThreadPool()
        self._thumb_pool.setMaxThreadCount(4)
        self._thumb_extracted.connect(
            self._on_source_thumb_extracted, Qt.ConnectionType.QueuedConnection)
        self._source_info_dialog: Optional[SourceInfoDialog] = None

        # Orphan-source registry: when the user removes a source from the Media
        # Pool but keeps its clips on the timeline, we record (file_path -> id).
        # On a later re-import of the same path, we reuse that id so the clips
        # bind back automatically. Persisted in the .psynth file via project.py.
        self._orphan_paths: Dict[str, str] = {}

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_clips = QLabel("0 clips")
        self._status_duration = QLabel("00:00")
        self._status_frame = QLabel("Frame 0")
        self._status_bar.addWidget(self._status_clips)
        self._status_bar.addWidget(self._status_duration)
        self._status_bar.addPermanentWidget(self._status_frame)

        self._timeline.clips_changed.connect(self._update_status)
        self._timeline.clips_changed.connect(self._mark_dirty)
        self._timeline.in_out_changed.connect(self._mark_dirty)

        # Menu bar
        self._setup_menus()
        self._update_title()

        # Timeline-specific keys (arrows / Home / End / Up / Down) are
        # QShortcuts scoped to the timeline widget so they only fire when
        # it has focus. Customizable via the Keyboard Shortcuts dialog.
        self._setup_timeline_shortcuts()

        # Ctrl+Shift+D — diagnostic thread-stack dump to src/diag.log.
        # Use during a hang to capture every Python thread's current stack.
        # Cheap and side-effect-free; safe to leave enabled.
        diag_action = QAction("Dump diagnostic threads", self)
        diag_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        diag_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        diag_action.triggered.connect(self._dump_diag)
        self.addAction(diag_action)

        # App-wide capture of mouse Back/Forward buttons for undo/redo,
        # so a mouse-driven workflow doesn't have to bounce to the keyboard.
        QApplication.instance().installEventFilter(self)

    def _setup_timeline_shortcuts(self):
        """Wire the timeline-widget keys (arrow stepping, Home/End, clip
        navigation) as QShortcuts scoped to the timeline so they only fire
        when it has focus. Each is registered with the ShortcutManager so
        the Keyboard Shortcuts dialog can rebind them."""
        from PySide6.QtGui import QShortcut
        strip = self._timeline_widget.strip
        pairs = [
            ("playhead_left",       lambda: strip.step_playhead(-1)),
            ("playhead_left_fast",  lambda: strip.step_playhead(-10)),
            ("playhead_right",      lambda: strip.step_playhead(1)),
            ("playhead_right_fast", lambda: strip.step_playhead(10)),
            ("playhead_home",       strip.go_to_start),
            ("playhead_end",        strip.go_to_end),
            ("select_next_clip",    lambda: strip.select_adjacent_clip(1)),
            ("select_prev_clip",    lambda: strip.select_adjacent_clip(-1)),
        ]
        for sid, handler in pairs:
            sc = QShortcut(QKeySequence(), self._timeline_widget)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(handler)
            self._shortcut_mgr.attach_qshortcut(sid, sc)

        # People — 10 digit shortcuts toggle the selected clips' membership
        # in the group bound to that digit.
        for digit in range(10):
            sc = QShortcut(QKeySequence(), self._timeline_widget)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(
                lambda d=digit: self._on_group_digit(d))
            self._shortcut_mgr.attach_qshortcut(f"group_digit_{digit}", sc)

    def _on_group_digit(self, digit: int):
        """Toggle the digit-bound group on every selected clip. If no group
        owns this digit yet, prompt for a name and create one inline."""
        selected = [cid for cid in self._timeline.selected_ids
                    if (c := self._timeline.get_clip_by_id(cid))
                    and not c.is_gap]
        if not selected:
            return
        group = self._timeline.get_group_by_digit(digit)
        if group is None:
            name, ok = QInputDialog.getText(
                self, "Create Group",
                f"Name for new group bound to digit {digit}:")
            if not ok:
                return
            name = name.strip()
            if not name:
                return
            group = self._timeline.add_group(name=name, digit=digit)
        self._timeline.toggle_clip_group(selected, group.id)

    def _on_keyboard_shortcuts(self):
        """File → Keyboard Shortcuts… opens the rebinding dialog."""
        from ui.shortcuts_dialog import KeyboardShortcutsDialog
        dlg = KeyboardShortcutsDialog(self._shortcut_mgr, parent=self)
        dlg.exec()

    def _dump_diag(self):
        """Diagnostic shortcut handler — writes a thread dump to diag.log."""
        from utils.diag import diag, dump_all_thread_stacks
        diag("[user] Ctrl+Shift+D pressed, dumping thread stacks")
        dump_all_thread_stacks(reason="user pressed Ctrl+Shift+D")
        try:
            self._status_bar.showMessage(
                "Diagnostic thread dump written to src/diag.log", 4000)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            btn = event.button()
            if btn == Qt.MouseButton.BackButton:
                self._timeline.undo()
                return True
            if btn == Qt.MouseButton.ForwardButton:
                self._timeline.redo()
                return True
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        super().showEvent(event)
        # Init mpv after widget is shown (needs valid winId)
        self._preview.init_player()

    def _setup_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        new_action = QAction(icon("new"), "New Project", self)
        self._shortcut_mgr.attach_action("new_project", new_action)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction(icon("open"), "Open Project...", self)
        self._shortcut_mgr.attach_action("open_project", open_action)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        save_action = QAction(icon("save"), "Save Project", self)
        self._shortcut_mgr.attach_action("save_project", save_action)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction(icon("save-as"), "Save Project As...", self)
        self._shortcut_mgr.attach_action("save_as", save_as_action)
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        shortcuts_action = QAction("Keyboard Shortcuts...", self)
        shortcuts_action.triggered.connect(self._on_keyboard_shortcuts)
        file_menu.addAction(shortcuts_action)

        edit_menu = menu.addMenu("Edit")

        split_action = QAction(icon("scissors"), "Split at Playhead", self)
        self._shortcut_mgr.attach_action("split", split_action)
        split_action.triggered.connect(self._on_split)
        edit_menu.addAction(split_action)

        delete_action = QAction(icon("trash"), "Delete Selected", self)
        self._shortcut_mgr.attach_action("delete", delete_action)
        delete_action.triggered.connect(self._on_delete)
        edit_menu.addAction(delete_action)

        ripple_delete_action = QAction(icon("ripple-delete"), "Ripple Delete Selected", self)
        self._shortcut_mgr.attach_action("ripple_delete", ripple_delete_action)
        ripple_delete_action.triggered.connect(self._on_ripple_delete)
        edit_menu.addAction(ripple_delete_action)

        edit_menu.addSeparator()

        undo_action = QAction(icon("undo"), "Undo", self)
        self._shortcut_mgr.attach_action("undo", undo_action)
        undo_action.triggered.connect(self._timeline.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction(icon("redo"), "Redo", self)
        self._shortcut_mgr.attach_action("redo", redo_action)
        redo_action.triggered.connect(self._timeline.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction(icon("select-all"), "Select All", self)
        self._shortcut_mgr.attach_action("select_all", select_all_action)
        select_all_action.triggered.connect(self._on_select_all)
        edit_menu.addAction(select_all_action)

        select_gap_action = QAction(icon("select-gap"), "Select to Gap Left", self)
        self._shortcut_mgr.attach_action("select_to_gap", select_gap_action)
        select_gap_action.triggered.connect(self._on_select_to_gap)
        edit_menu.addAction(select_gap_action)

        edit_menu.addSeparator()

        set_in_action = QAction(icon("set-in"), "Set In Point", self)
        self._shortcut_mgr.attach_action("set_in", set_in_action)
        set_in_action.triggered.connect(self._on_set_in_point)
        edit_menu.addAction(set_in_action)

        set_out_action = QAction(icon("set-out"), "Set Out Point", self)
        self._shortcut_mgr.attach_action("set_out", set_out_action)
        set_out_action.triggered.connect(self._on_set_out_point)
        edit_menu.addAction(set_out_action)

        clear_in_out_action = QAction(icon("clear-in-out"), "Clear In/Out", self)
        self._shortcut_mgr.attach_action("clear_in_out", clear_in_out_action)
        clear_in_out_action.triggered.connect(self._on_clear_in_out)
        edit_menu.addAction(clear_in_out_action)

        edit_menu.addSeparator()

        selection_mode_action = QAction(icon("cursor"), "Selection Mode", self)
        self._shortcut_mgr.attach_action("selection_mode", selection_mode_action)
        selection_mode_action.triggered.connect(lambda: self._set_edit_mode(EditMode.SELECTION))
        edit_menu.addAction(selection_mode_action)

        cut_mode_action = QAction(icon("cut-mode"), "Cut Mode", self)
        self._shortcut_mgr.attach_action("cut_mode", cut_mode_action)
        cut_mode_action.triggered.connect(lambda: self._set_edit_mode(EditMode.CUT))
        edit_menu.addAction(cut_mode_action)

        scrub_follow_action = QAction(icon("target"), "Scrub Follow", self)
        self._shortcut_mgr.attach_action("scrub_follow", scrub_follow_action)
        scrub_follow_action.triggered.connect(self._toggle_scrub_follow)
        edit_menu.addAction(scrub_follow_action)

        # --- View menu ---
        view_menu = menu.addMenu("View")
        scale_menu = view_menu.addMenu("UI Scale")
        self._scale_action_group = QActionGroup(self)
        self._scale_action_group.setExclusive(True)
        current_pct = ui_scale().percent
        from core.ui_scale import UIScale
        for pct in UIScale.PRESETS:
            act = QAction(f"{pct}%", self)
            act.setCheckable(True)
            if pct == current_pct:
                act.setChecked(True)
            act.triggered.connect(
                lambda _checked=False, p=pct: ui_scale().set_percent(p))
            self._scale_action_group.addAction(act)
            scale_menu.addAction(act)
        self._scale_actions = {
            int(a.text().rstrip("%")): a
            for a in self._scale_action_group.actions()
        }

        # --- Timeline menu ---
        timeline_menu = menu.addMenu("Timeline")

        import_action = QAction(icon("import"), "Import Video...", self)
        self._shortcut_mgr.attach_action("import_video", import_action)
        import_action.triggered.connect(self._on_import)
        timeline_menu.addAction(import_action)

        timeline_menu.addSeparator()

        export_vid_action = QAction(icon("film"), "Export Video...", self)
        self._shortcut_mgr.attach_action("export_video", export_vid_action)
        export_vid_action.triggered.connect(self._on_export_video)
        timeline_menu.addAction(export_vid_action)

        export_img_action = QAction(icon("image"), "Export Image Sequence...", self)
        self._shortcut_mgr.attach_action("export_images", export_img_action)
        export_img_action.triggered.connect(self._on_export_images)
        timeline_menu.addAction(export_img_action)

        export_audio_action = QAction(icon("audio"), "Export Audio Only...", self)
        export_audio_action.triggered.connect(self._on_export_audio)
        timeline_menu.addAction(export_audio_action)

        xml_action = QAction(icon("document"), "Export XML (FCPXML)...", self)
        xml_action.triggered.connect(self._on_export_xml)
        timeline_menu.addAction(xml_action)

        otio_action = QAction(icon("document"), "Export OTIO...", self)
        otio_action.triggered.connect(self._on_export_otio)
        timeline_menu.addAction(otio_action)

        timeline_menu.addSeparator()

        detect_action = QAction(icon("wand"), "Detect Cuts...", self)
        self._shortcut_mgr.attach_action("detect_cuts", detect_action)
        detect_action.triggered.connect(self._on_detect_cuts)
        timeline_menu.addAction(detect_action)

        timeline_menu.addSeparator()

        play_action = QAction(icon("play"), "Play/Pause", self)
        self._shortcut_mgr.attach_action("play_pause", play_action)
        play_action.triggered.connect(self._toggle_play)
        timeline_menu.addAction(play_action)

        # --- Tools menu ---
        tools_menu = menu.addMenu("Tools")

        relink_action = QAction(icon("link"), "Relink…", self)
        relink_action.triggered.connect(self._on_tools_relink)
        tools_menu.addAction(relink_action)

        cache_action = QAction(icon("cache"), "Cache Manager…", self)
        cache_action.triggered.connect(self._on_tools_cache_manager)
        tools_menu.addAction(cache_action)

        cut_inspect_action = QAction(icon("cut-inspect"), "Cut Inspect…", self)
        cut_inspect_action.triggered.connect(self._on_tools_cut_inspect)
        tools_menu.addAction(cut_inspect_action)

    # --- Import ---

    _VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mxf'}

    def _get_timeline_ref(self):
        """Return (width, height, fps) from the first existing source, or zeros."""
        if self._sources:
            s = next(iter(self._sources.values()))
            return s.width, s.height, s.fps
        return 0, 0, 0.0

    def _on_import(self):
        ref_w, ref_h, ref_fps = self._get_timeline_ref()
        dialog = ImportDialog(ref_width=ref_w, ref_height=ref_h,
                              ref_fps=ref_fps, parent=self)
        dialog.import_complete.connect(self._on_import_complete)
        dialog.exec()

    def _import_files(self, file_paths: list):
        """Import video files directly (used by drag-and-drop and Media Panel
        drops). Probes each file, validates resolution/FPS against existing
        sources, and adds them to the media pool. Does NOT create clips."""
        from utils.ffprobe import probe_video
        ref_w, ref_h, ref_fps = self._get_timeline_ref()

        probed = []
        for path in file_paths:
            info = probe_video(path)
            if info is None:
                logger.warning("Could not probe dropped file: %s", path)
                return
            probed.append((path, info))

        # Use first file as reference if timeline/pool is empty
        if ref_w == 0 and probed:
            ref_w, ref_h, ref_fps = probed[0][1].width, probed[0][1].height, probed[0][1].fps

        # Validate all files match reference
        for path, info in probed:
            name = os.path.basename(path)
            if info.width != ref_w or info.height != ref_h:
                QMessageBox.critical(
                    self, "Import Error",
                    f"Resolution mismatch — batch rejected.\n\n"
                    f"{name} is {info.width}x{info.height}, "
                    f"expected {ref_w}x{ref_h}"
                )
                return
            if abs(info.fps - ref_fps) > 0.02:
                QMessageBox.critical(
                    self, "Import Error",
                    f"FPS mismatch — batch rejected.\n\n"
                    f"{name} is {info.fps:.3f} fps, "
                    f"expected {ref_fps:.3f} fps"
                )
                return

        sources = [VideoSource(
            file_path=path,
            total_frames=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            codec=info.codec,
            audio_codec=info.audio_codec,
            audio_sample_rate=info.audio_sample_rate,
            audio_channels=info.audio_channels,
            time_base_num=info.time_base_num,
            time_base_den=info.time_base_den,
        ) for path, info in probed]

        self._on_import_complete(sources)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in self._VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in self._VIDEO_EXTENSIONS:
                    paths.append(path)
        if paths:
            self._import_files(paths)

    def _on_import_complete(self, sources: list):
        """Add imported sources to the Media Pool. Does NOT create timeline
        clips — the user drags from the pool to the timeline when ready.
        If the file path matches a previously-removed source whose clips were
        kept on the timeline, the new source reuses the old UUID so those
        clips bind back automatically."""
        if not sources:
            return
        for source in sources:
            # Orphan revival: exact file_path match against the removed-paths
            # registry. The new source takes the old UUID so existing clips
            # (still referencing it) immediately become valid again.
            old_id = self._orphan_paths.pop(source.file_path, None)
            if old_id is not None and old_id not in self._sources:
                logger.info("Reviving orphan source %s -> %s for %s",
                             source.id, old_id, source.file_path)
                source.id = old_id
            self._sources[source.id] = source
            self._reader_pool.register_source(source)
            self._proxy_manager.load_or_open(source)
            self._thumb_pool.start(_SourceThumbWorker(source, self._thumb_extracted))

        # Set FPS from the first imported source if the timeline FPS isn't set yet
        if sources:
            self._timeline_widget.set_fps(sources[0].fps)

        self._refresh_media_panel()
        self._dirty = True
        self._update_status()
        for source in sources:
            logger.info("Imported %s", source.file_path)

        self._warn_unsafe_timebases(sources)

    def _warn_unsafe_timebases(self, sources: list):
        """Inform the user, once per import / relink / project load, when
        any source's container time_base can't exactly represent its
        declared fps. Such sources cause ±1 frame drift on FCPXML / OTIO
        import into NLEs that time-seek the source (Resolve, Premiere).

        The dialog offers an in-app Auto-fix that runs ffmpeg ``-c copy
        -video_track_timescale <N>`` against each unsafe source and
        relinks the project to the produced ``*_fixed.mov``. ``-c copy``
        preserves frame count + ordering so all existing clip edits stay
        aligned with the new file.
        """
        unsafe = [s for s in sources if not s.is_seek_safe()]
        if not unsafe:
            return
        from ui.timebase_warning_dialog import TimebaseWarningDialog
        dlg = TimebaseWarningDialog(unsafe, parent=self)
        dlg.auto_fix_requested.connect(self._run_timebase_autofix)
        dlg.exec()

    def _run_timebase_autofix(self, jobs: list, audio_mode: str = "keep"):
        """jobs: list of (source_id, input_path, output_path,
        target_timescale, duration_seconds). audio_mode: one of the
        ``RemuxJob.AUDIO_*`` constants — picked by the user in the
        warning dialog's audio-handling selector. Spawns a modal progress
        dialog that drives a sequential ffmpeg remux and relinks each
        fixed file as it lands.
        """
        if not jobs:
            return
        from ui.remux_progress_dialog import RemuxProgressDialog
        progress = RemuxProgressDialog(jobs, audio_mode=audio_mode, parent=self)
        progress.source_succeeded.connect(self._apply_single_remux_relink)
        progress.exec()

    def _apply_single_remux_relink(self, source_id: str, fixed_path: str):
        """Pipe one freshly-remuxed file through the existing relink
        machinery (probe + _apply_relink_results). The relink updates
        VideoSource.file_path + new time_base, re-registers reader/proxy,
        refreshes thumbnails, and re-fires the timebase warning — which
        now sees the source as safe and stays silent.
        """
        from utils.ffprobe import probe_video
        info = probe_video(fixed_path)
        if info is None:
            QMessageBox.warning(
                self, "Relink failed",
                f"Remuxed file probed empty:\n{fixed_path}\n\n"
                "The fix-file was created but PrismaSynth couldn't read "
                "it back. Try Tools → Relink… manually.")
            return
        self._apply_relink_results({source_id: fixed_path},
                                   {source_id: info})

    # --- Media Panel handlers ---

    def _refresh_media_panel(self):
        self._media_panel.set_sources(self._sources)

    def _on_source_thumb_extracted(self, source_id: str):
        src = self._sources.get(source_id)
        if src is not None:
            self._media_panel.refresh_thumbnail(src)

    def _on_source_double_clicked(self, source_id: str):
        src = self._sources.get(source_id)
        if not src:
            return
        if self._source_info_dialog is None:
            self._source_info_dialog = SourceInfoDialog(self)
        self._source_info_dialog.show_for(src)

    def _on_source_relink_requested(self, source_ids):
        """Open RelinkDialog for the given source IDs (one or many)."""
        # Tolerate the legacy single-id call shape from older signal wiring.
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        sources = {sid: self._sources[sid] for sid in source_ids
                   if sid in self._sources}
        if not sources:
            return
        dlg = RelinkDialog(sources, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._apply_relink_results(dlg.resolved_paths(), dlg.probe_cache())

    def _on_select_all(self):
        """Ctrl+A handler. Routes by current keyboard focus: when the
        Media Pool (or any of its children) has focus, select all sources
        there; otherwise fall through to the timeline's select-all."""
        fw = self.focusWidget()
        if fw is not None and self._media_panel.isAncestorOf(fw):
            self._media_panel.select_all_sources()
            return
        self._timeline.select_all()

    def _on_tools_relink(self):
        """Tools → Relink… shows every imported source so the user can repoint
        any of them (missing or linked). The single-source flow stays on the
        Media Pool's right-click menu."""
        if not self._sources:
            QMessageBox.information(
                self, "Relink",
                "No sources to relink. Import a video first.")
            return
        dlg = RelinkDialog(self._sources, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._apply_relink_results(dlg.resolved_paths(), dlg.probe_cache())

    def _on_tools_cache_manager(self):
        """Tools → Cache Manager… opens the disk-cache management dialog."""
        from ui.cache_manager_dialog import CacheManagerDialog
        dlg = CacheManagerDialog(proxy_manager=self._proxy_manager, parent=self)
        dlg.exec()

    def _on_tools_cut_inspect(self):
        """Tools → Cut Inspect… launches scripts/cut_inspect.py as a separate
        Python process so its mpv instance is fully isolated from
        PrismaSynth's preview player. Two mpv instances in one process can
        deadlock or crash."""
        import subprocess
        import sys as _sys
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parent.parent.parent
        script = repo_root / "scripts" / "cut_inspect.py"
        if not script.exists():
            QMessageBox.warning(
                self, "Cut Inspect",
                f"Cut Inspect script not found:\n{script}")
            return
        flags = 0
        if _sys.platform == "win32":
            flags = (subprocess.DETACHED_PROCESS
                     | subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            subprocess.Popen(
                [_sys.executable, str(script)],
                cwd=str(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
        except OSError as e:
            QMessageBox.warning(
                self, "Cut Inspect",
                f"Could not launch Cut Inspect:\n{e}")

    def _apply_relink_results(self, resolved: dict, probe_cache: dict):
        """Apply the user's relink choices from RelinkDialog to one or more
        sources. Updates VideoSource metadata, clamps any clips whose out-point
        exceeds the new (shorter) source length, re-registers reader+proxy, and
        refreshes the source thumbnail. Shared by per-source relink (Media
        Pool) and the Tools menu's bulk-relink entry."""
        clamp_total = 0
        touched_sources: list = []
        from core.source_thumbnail import extract_thumbnail
        for source_id, new_path in resolved.items():
            if not new_path:
                continue  # user skipped this row
            src = self._sources.get(source_id)
            if not src:
                continue
            old_total_frames = src.total_frames
            src.file_path = new_path
            info = probe_cache.get(source_id)
            if info is not None:
                src.width = info.width
                src.height = info.height
                src.fps = info.fps
                src.total_frames = info.total_frames
                src.codec = info.codec
                src.audio_codec = info.audio_codec
                src.audio_sample_rate = info.audio_sample_rate
                src.audio_channels = info.audio_channels
                src.time_base_num = info.time_base_num
                src.time_base_den = info.time_base_den
                if info.total_frames < old_total_frames:
                    new_max = info.total_frames - 1
                    for c in self._timeline.clips:
                        if c.source_id == source_id:
                            if c.source_in > new_max:
                                c.source_in = new_max
                                c.source_out = new_max
                                clamp_total += 1
                            elif c.source_out > new_max:
                                c.source_out = new_max
                                clamp_total += 1
            self._reader_pool.register_source(src)
            self._proxy_manager.load_or_open(src, force_reopen=True)
            extract_thumbnail(src, force=True)
            self._media_panel.refresh_thumbnail(src)
            touched_sources.append(src)
        if clamp_total:
            QMessageBox.warning(
                self, "Clips Clamped",
                f"{clamp_total} clip(s) had their out-points trimmed because "
                "a relinked source has fewer frames.",
            )
        if touched_sources:
            self._dirty = True
            self._update_status()
            self._timeline_widget.update()
            self._warn_unsafe_timebases(touched_sources)

    def _on_source_remove_requested(self, source_ids):
        """Remove one or many sources from the Media Pool. Single-source
        keeps the original confirm flow; multi-source aggregates the clip
        count and shows one batch prompt."""
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        valid = [sid for sid in source_ids if sid in self._sources]
        if not valid:
            return
        if len(valid) == 1:
            self._remove_single_source(valid[0])
        else:
            self._remove_multi_sources(valid)

    def _remove_single_source(self, source_id: str):
        src = self._sources.get(source_id)
        if not src:
            return
        from pathlib import Path as _Path
        name = _Path(src.file_path).name
        clip_count = self._timeline.count_clips_for_source(source_id)
        if clip_count == 0:
            if QMessageBox.question(
                self, "Remove Source",
                f"Remove '{name}' from the media pool?",
            ) != QMessageBox.StandardButton.Yes:
                return
            self._do_remove_source(source_id, remove_clips=False)
            return
        dlg = RemoveSourceDialog(name, clip_count, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        if dlg.action == RemoveSourceAction.REMOVE_WITH_CLIPS:
            self._do_remove_source(source_id, remove_clips=True)
        elif dlg.action == RemoveSourceAction.REMOVE_KEEP_CLIPS:
            self._do_remove_source(source_id, remove_clips=False)

    def _remove_multi_sources(self, source_ids: list):
        n = len(source_ids)
        total_clips = sum(
            self._timeline.count_clips_for_source(sid) for sid in source_ids)
        if total_clips == 0:
            if QMessageBox.question(
                self, "Remove Sources",
                f"Remove {n} sources from the media pool?",
            ) != QMessageBox.StandardButton.Yes:
                return
            for sid in source_ids:
                self._do_remove_source(sid, remove_clips=False)
            return
        # source_name is unused when source_count > 1 (the dialog formats
        # its own message), but the parameter is positional.
        dlg = RemoveSourceDialog("", total_clips,
                                  source_count=n, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        if dlg.action == RemoveSourceAction.REMOVE_WITH_CLIPS:
            for sid in source_ids:
                self._do_remove_source(sid, remove_clips=True)
        elif dlg.action == RemoveSourceAction.REMOVE_KEEP_CLIPS:
            for sid in source_ids:
                self._do_remove_source(sid, remove_clips=False)

    def _do_remove_source(self, source_id: str, *, remove_clips: bool):
        src = self._sources.get(source_id)
        if remove_clips:
            self._timeline.remove_source_clips(source_id)
        elif src is not None and src.file_path:
            # Track this source for potential revival on a same-path re-import.
            self._orphan_paths[src.file_path] = source_id
        # Drop the source from all bookkeeping. Orphaned clips (when
        # remove_clips=False) render black via the missing-source path.
        self._sources.pop(source_id, None)
        try:
            self._reader_pool.unregister_source(source_id)
        except Exception:
            # Older reader pool may not have unregister; best-effort cleanup.
            pass
        self._refresh_media_panel()
        self._dirty = True
        self._update_status()

    def _on_top_splitter_moved(self, *_):
        QSettings().setValue(
            "main_window/top_splitter_sizes", self._top_splitter.sizes())

    def _on_main_splitter_moved(self, *_):
        QSettings().setValue(
            "main_window/vertical_splitter_sizes", self._main_splitter.sizes())

    def _on_right_tab_changed(self, idx: int):
        QSettings().setValue("main_window/right_tab_index", int(idx))

    def _on_ui_scale_changed(self):
        """Re-apply scaled chrome sizes when the user picks a new UI scale.
        Splitter sizes and the user's draggable clip-height are intentionally
        left absolute (they reflect deliberate user gestures). Persistent
        panels manage their own re-apply via their own UIScale.changed slots."""
        s = ui_scale()
        # Top separator + splitter handle (visual chrome only).
        try:
            self._top_separator.setFixedHeight(s.px(4))
        except AttributeError:
            pass
        try:
            self._top_splitter.setHandleWidth(s.px(4))
        except AttributeError:
            pass
        # Timeline container's left/right margin scales with the rest of the chrome.
        try:
            m = s.px(16)
            self._timeline_container_layout.setContentsMargins(m, 0, m, 0)
        except AttributeError:
            pass
        # Re-check the matching menu radio item (handles programmatic changes).
        act = self._scale_actions.get(s.percent) if hasattr(self, "_scale_actions") else None
        if act is not None and not act.isChecked():
            act.setChecked(True)

    # --- Timeline drop handlers (from Media Panel drag) ---

    def _on_timeline_sources_dropped(self, source_ids: list, frame: int):
        """Insert one new whole-source clip per dragged source at the drop frame."""
        clips = []
        for sid in source_ids:
            src = self._sources.get(sid)
            if src is None or src.total_frames <= 0:
                continue
            clips.append(Clip(
                source_id=src.id,
                source_in=0,
                source_out=src.total_frames - 1,
            ))
        if not clips:
            return
        self._timeline.insert_clips_at_frame(clips, frame)
        # Set FPS from the first source if not already set
        first_src = self._sources.get(source_ids[0])
        if first_src is not None:
            self._timeline_widget.set_fps(first_src.fps)
        self._start_thumbnail_cache()
        self._dirty = True
        self._update_status()

    def _on_timeline_files_dropped(self, paths: list, frame: int):
        """Files dropped directly on the timeline: import to pool AND insert
        clips at the drop frame."""
        # Snapshot existing source IDs so we can identify the freshly-added ones.
        prior_ids = set(self._sources.keys())
        self._import_files(paths)
        new_ids = [sid for sid in self._sources.keys() if sid not in prior_ids]
        if not new_ids:
            return  # validation rejected the import
        self._on_timeline_sources_dropped(new_ids, frame)

    def _on_detect_cuts(self):
        """Run cut detection on all non-gap clips on the timeline.
        Respects in/out render range when set."""
        if not self._sources:
            return

        # Pause GPU-heavy subsystems to avoid CUDA/mpv contention during detection
        self._preview.pause()
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.pause()

        # Determine analysis range
        in_out_limited = (self._timeline.in_point is not None
                          or self._timeline.out_point is not None)
        if in_out_limited:
            r_start, r_end = self._timeline.get_render_range()
        else:
            r_start = 0
            r_end = self._timeline.get_total_duration_frames() - 1

        # Build segments from timeline clips within the analysis range
        segments = []
        # Track prefix/suffix clips for partially-clamped clips
        self._detect_partials = {}  # clip_id -> (prefix_Clip|None, suffix_Clip|None)
        pos = 0
        for clip in self._timeline.clips:
            clip_start = pos
            clip_end = pos + clip.duration_frames - 1
            pos += clip.duration_frames

            if clip.is_gap:
                continue
            # Skip clips entirely outside the analysis range
            if clip_end < r_start or clip_start > r_end:
                continue

            # Clamp clip to the analysis range
            clamp_start = max(clip_start, r_start)
            clamp_end = min(clip_end, r_end)
            offset_start = clamp_start - clip_start
            offset_end = clamp_end - clip_start
            source_in = clip.source_in + offset_start
            source_out = clip.source_in + offset_end

            # Build prefix/suffix for partially-clamped clips
            prefix = None
            suffix = None
            if offset_start > 0:
                prefix = Clip(source_id=clip.source_id,
                              source_in=clip.source_in,
                              source_out=source_in - 1,
                              color_index=clip.color_index)
            if source_out < clip.source_out:
                suffix = Clip(source_id=clip.source_id,
                              source_in=source_out + 1,
                              source_out=clip.source_out,
                              color_index=clip.color_index)
            if prefix or suffix:
                self._detect_partials[clip.id] = (prefix, suffix)

            segments.append((clip.source_id, source_in, source_out, clip.id))

        if not segments:
            if self._thumbnail_cache is not None:
                self._thumbnail_cache.resume()
            return

        dialog = DetectDialog(segments, self._sources,
                              in_out_limited=in_out_limited, parent=self)
        dialog.detection_complete.connect(self._on_detection_complete)
        dialog.exec()

        # Clean up partials regardless of accept/reject
        self._detect_partials = {}

        # Resume thumbnail generation after dialog closes
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.resume()

    def _on_detection_complete(self, results: dict):
        """Replace analyzed clips with detected scene clips.
        results: {clip_id: [Clip, ...]}"""
        # Re-attach prefix/suffix clips for partially-clamped clips
        partials = self._detect_partials
        for clip_id, sub_clips in results.items():
            if clip_id in partials:
                prefix, suffix = partials[clip_id]
                if prefix:
                    sub_clips.insert(0, prefix)
                if suffix:
                    sub_clips.append(suffix)
        self._detect_partials = {}

        self._timeline.replace_detected(results)

        # Reload proxies saved by scene detector (force reopen to pick up new data)
        seen_sources = set()
        for clips in results.values():
            for clip in clips:
                if clip.source_id and clip.source_id not in seen_sources:
                    seen_sources.add(clip.source_id)
                    source = self._sources.get(clip.source_id)
                    if source:
                        self._proxy_manager.load_or_open(source, force_reopen=True)

        self._start_thumbnail_cache()

        total_clips = sum(len(v) for v in results.values())
        self._update_status()
        logger.info("Detection complete: %d clips from %d segments",
                     total_clips, len(results))

    def _start_thumbnail_cache(self):
        # Respect the user's toggle — master off means nothing runs.
        if not self._timeline_widget.thumbnails_enabled:
            return
        visible = set(self._timeline_widget.strip.get_visible_clip_ids())
        playhead = self._timeline_widget.strip.playhead_frame
        if self._thumbnail_cache is None:
            self._thumbnail_cache = ThumbnailCache(
                self._timeline, self._sources,
                proxy_manager=self._proxy_manager,
            )
            self._thumbnail_cache.thumbnail_ready.connect(
                self._on_thumbnail_ready, Qt.ConnectionType.QueuedConnection)
            self._thumbnail_cache.set_hq_enabled(self._timeline_widget.hq_thumbnails_enabled)
            self._thumbnail_cache.start(
                priority_clip_ids=visible, playhead_frame=playhead)
        else:
            self._thumbnail_cache.notify_clips_changed()
            self._thumbnail_cache.reprioritize(visible, playhead)

    def _on_thumbnails_toggled(self, enabled: bool):
        if enabled:
            self._start_thumbnail_cache()
        elif self._thumbnail_cache is not None:
            self._thumbnail_cache.stop()
            self._thumbnail_cache = None

    def _on_hq_thumbnails_toggled(self, enabled: bool):
        # Master off → advanced toggle has no immediate effect; state is
        # applied next time the cache is started.
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.set_hq_enabled(enabled)

    def _on_cache_thumbnails_clicked(self):
        """Toolbar/timeline button: open the modal Cache Thumbnails dialog.
        The dialog drives the bulk job lifecycle and blocks the main window
        until it's closed."""
        if not self._timeline.clips:
            self._status_bar.showMessage("No clips to cache", 3000)
            return
        # Master toggle off → start the cache so disk writes still emit.
        if self._thumbnail_cache is None:
            self._start_thumbnail_cache()
        if self._thumbnail_cache is None:
            return
        from ui.cache_thumbnails_dialog import CacheThumbnailsDialog
        render_range = self._timeline.get_render_range()
        has_in_out = (self._timeline.in_point is not None
                      or self._timeline.out_point is not None)
        dialog = CacheThumbnailsDialog(
            self._thumbnail_cache, render_range, has_in_out, self)
        dialog.exec()

    def _on_thumbnail_ready(self, clip_id: str, position: str, qimage):
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap.fromImage(qimage)
        self._timeline_widget.strip.set_thumbnail(clip_id, position, pixmap)

    def _on_viewport_changed(self):
        """Scroll or zoom changed — debounce before reprioritizing thumbnails."""
        self._viewport_timer.start()

    def _do_reprioritize(self):
        """Actually reprioritize after scroll settles (300ms debounce)."""
        if self._thumbnail_cache:
            visible = set(self._timeline_widget.strip.get_visible_clip_ids())
            playhead = self._timeline_widget.strip.playhead_frame
            self._thumbnail_cache.reprioritize(visible, playhead)

    # --- Playhead ---

    def _on_playhead_changed(self, frame: int):
        # Playback-driven update — just sync UI, don't touch mpv
        if self._playback_updating:
            self._status_frame.setText(f"Frame {frame}")
            return

        # User moved the playhead — restart playback from new position if playing
        if self._preview.is_playing or self._playback_timer.isActive():
            self._stop_playback()
            self._timeline_widget.strip._playhead_frame = frame
            self._start_playback()
            return

        self._status_frame.setText(f"Frame {frame}")

        result = self._timeline.get_clip_at_position(frame)

        if self._selection_follows_playhead and result:
            clip, _ = result
            self._timeline.select_clip(clip.id)

        # EXPERIMENT v0.9.x: don't pause the thumbnail coordinator during
        # scrub. With a full pre-bake on disk, the coordinator's work is
        # cheap (JPEG decode) and shouldn't contend with mpv. Revert if
        # playhead drag feels laggy.

        # Seek mpv to the correct source frame (GPU-accelerated)
        if result is None or result[0].is_gap:
            self._preview.show_black()
            return
        clip, offset = result
        source = self._sources.get(clip.source_id)
        if source:
            source_frame = clip.source_in + offset
            self._preview.load_source(source.file_path)
            self._preview.seek_to_frame(source_frame, source.fps)

    def _resume_thumbnails(self):
        if self._thumbnail_cache:
            self._thumbnail_cache.resume()

    def _on_timeline_pan_started(self):
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.pause()

    def _on_timeline_pan_ended(self):
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.resume()

    # --- Clip selection ---

    def _on_clip_clicked(self, clip_id: str, event):
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._timeline.toggle_select(clip_id)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            if self._last_clicked_clip_id:
                self._timeline.select_range(self._last_clicked_clip_id, clip_id)
            else:
                self._timeline.select_clip(clip_id)
        else:
            self._timeline.select_clip(clip_id)
        self._last_clicked_clip_id = clip_id

        # Move playhead to start of clicked clip
        start = self._timeline.get_clip_timeline_start(clip_id)
        if start >= 0:
            self._timeline_widget.set_playhead(start)

    def _on_selection_changed(self):
        selected = self._timeline.get_selected_clips()
        if len(selected) == 1:
            self._clip_info.update_clip(selected[0], self._sources)
        elif len(selected) == 0:
            self._clip_info.update_clip(None, self._sources)
        else:
            # Multiple selected — show count
            self._clip_info.update_clip(selected[0], self._sources)

    def _on_selection_follows_toggled(self, checked: bool):
        self._selection_follows_playhead = checked

    # --- Editing ---

    def _on_split(self):
        frame = self._timeline_widget.strip.playhead_frame
        result = self._timeline.get_clip_at_position(frame)
        if result:
            clip, _ = result
            if clip.is_gap:
                return  # can't split a gap
            self._timeline.split_clip_at(clip.id, frame)
            self._start_thumbnail_cache()

    def _earliest_selected_frame(self) -> int:
        """Get timeline position of the earliest selected clip."""
        selected = self._timeline.selected_ids
        if not selected:
            return -1
        earliest = -1
        for clip in self._timeline.clips:
            if clip.id in selected:
                pos = self._timeline.get_clip_timeline_start(clip.id)
                if earliest < 0 or pos < earliest:
                    earliest = pos
                break  # clips are ordered, first match is earliest
        return earliest

    def _on_delete(self):
        """Delete selected clips, replacing with gaps."""
        if not self._timeline.selected_ids:
            return
        # Teleport playhead only when deleting gaps
        selected = self._timeline.selected_ids
        deleting_gaps = all(
            c.is_gap for c in self._timeline.clips if c.id in selected)
        target_frame = self._earliest_selected_frame() if deleting_gaps else -1
        self._timeline.delete_selected()
        if target_frame >= 0:
            self._timeline_widget.set_playhead(target_frame)
            self._timeline_widget.strip.ensure_playhead_visible()
        self._refresh_preview_after_edit()

    def _on_ripple_delete(self):
        """Ripple delete selected clips — collapse the space."""
        if not self._timeline.selected_ids:
            return
        # Only teleport playhead when deleting gaps
        selected = self._timeline.selected_ids
        deleting_gaps = all(
            c.is_gap for c in self._timeline.clips if c.id in selected)
        target_frame = self._earliest_selected_frame() if deleting_gaps else -1
        self._timeline.ripple_delete_selected()
        if target_frame >= 0:
            total = self._timeline.get_total_duration_frames()
            target_frame = min(target_frame, max(0, total - 1))
            self._timeline_widget.set_playhead(target_frame)
            self._timeline_widget.strip.ensure_playhead_visible()
        self._refresh_preview_after_edit()

    def _refresh_preview_after_edit(self):
        """Refresh preview after a timeline edit (delete, ripple delete, etc.)."""
        frame = self._timeline_widget.strip.playhead_frame
        self._on_playhead_changed(frame)

    def _on_select_to_gap(self):
        frame = self._timeline_widget.strip.playhead_frame
        self._timeline.select_to_gap_left(frame)

    # --- In/Out Points ---

    def _on_set_in_point(self):
        frame = self._timeline_widget.strip.playhead_frame
        self._timeline.set_in_point(frame)

    def _on_set_out_point(self):
        frame = self._timeline_widget.strip.playhead_frame
        self._timeline.set_out_point(frame)

    def _on_clear_in_out(self):
        self._timeline.clear_in_out()

    def _get_render_frame_count(self) -> int:
        start, end = self._timeline.get_render_range()
        count = 0
        pos = 0
        for clip in self._timeline.clips:
            clip_start = pos
            clip_end = pos + clip.duration_frames - 1
            pos += clip.duration_frames
            if clip_end < start or clip_start > end:
                continue
            if clip.is_gap:
                continue
            effective_start = max(clip_start, start)
            effective_end = min(clip_end, end)
            count += effective_end - effective_start + 1
        return count

    # --- Edit Mode ---

    def _toggle_scrub_follow(self):
        active = self._timeline_widget.strip.toggle_scrub_follow()
        logger.info("Scrub follow: %s", "ON" if active else "OFF")

    def _set_edit_mode(self, mode: EditMode):
        self._timeline_widget.strip.set_edit_mode(mode)
        self._toolbar.set_mode(int(mode))

    def _on_mode_changed(self, mode: int):
        self._timeline_widget.strip.set_edit_mode(EditMode(mode))

    def _on_preview_frame_requested(self, frame: int):
        """Preview a frame without moving the playhead (cut mode hover scrub)."""
        # EXPERIMENT v0.9.x: same as _on_playhead_changed — coordinator
        # stays running during cut-mode hover. Revert if hover feels laggy.

        result = self._timeline.get_clip_at_position(frame)
        if result is None or result[0].is_gap:
            self._preview.show_black()
            return
        clip, offset = result
        source = self._sources.get(clip.source_id)
        if source:
            source_frame = clip.source_in + offset
            self._preview.load_source(source.file_path)
            self._preview.seek_to_frame(source_frame, source.fps)

    def _on_cut_at_frame(self, frame: int):
        """Handle a cut-mode click: split the clip at this frame, select left half."""
        result = self._timeline.get_clip_at_position(frame)
        if result is None:
            return
        clip, _ = result
        if clip.is_gap:
            return
        # Select the clip first so split_clip_at can transfer selection to left half
        self._timeline.select_clip(clip.id)
        if self._timeline.split_clip_at(clip.id, frame, select_left_only=True):
            self._start_thumbnail_cache()

    # --- Playback ---

    def _toggle_play(self):
        # _preview.is_playing reflects mpv's pause state, but during a gap
        # mpv stays paused while the wall-clock timer drives the playhead.
        # Check both so Space can stop playback regardless of which mode.
        if self._preview.is_playing or self._playback_timer.isActive():
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self._timeline.clip_count == 0:
            return
        frame = self._timeline_widget.strip.playhead_frame
        result = self._timeline.get_clip_at_position(frame)
        if result is None:
            return
        clip, offset = result

        if clip.is_gap:
            # Start on a gap: show black, advance playhead via timer at source fps
            self._preview.show_black()
            self._playback_source = None
            self._playback_clip = clip
            self._playback_clip_timeline_start = self._timeline.get_clip_timeline_start(clip.id)
            self._toolbar.set_playing(True)
            self._gap_start_time = time.monotonic()
            self._gap_start_frame = frame
            self._playback_timer.start()
        else:
            source = self._sources.get(clip.source_id)
            if not source:
                return
            self._preview.load_source(source.file_path)
            self._preview.seek_to_frame(clip.source_in + offset, source.fps)
            self._preview.play()
            self._toolbar.set_playing(True)
            self._playback_source = source
            self._playback_clip = clip
            self._playback_clip_timeline_start = self._timeline.get_clip_timeline_start(clip.id)
            self._gap_playback_frame = -1
            self._playback_timer.start()

    def _stop_playback(self):
        self._playback_timer.stop()
        self._preview.pause()
        self._toolbar.set_playing(False)

    def _on_playback_tick(self):
        """Sync timeline playhead with mpv's current position during native playback."""
        clip = self._playback_clip
        if clip is None:
            self._stop_playback()
            return

        # Gap playback: advance playhead based on elapsed time at source fps
        if clip.is_gap:
            fps = next(iter(self._sources.values())).fps if self._sources else 24.0
            elapsed = time.monotonic() - self._gap_start_time
            current_frame = self._gap_start_frame + int(elapsed * fps)
            gap_end = self._playback_clip_timeline_start + clip.duration_frames
            if current_frame >= gap_end:
                # Gap finished — move to next clip
                result = self._timeline.get_clip_at_position(gap_end)
                if result is None:
                    self._stop_playback()
                    return
                next_clip, _ = result
                if next_clip.is_gap:
                    # Another gap — keep going, reset timer
                    self._playback_clip = next_clip
                    self._playback_clip_timeline_start = gap_end
                    self._gap_start_time = time.monotonic()
                    self._gap_start_frame = gap_end
                else:
                    # Real clip — start mpv playback
                    next_source = self._sources.get(next_clip.source_id)
                    if not next_source:
                        self._stop_playback()
                        return
                    self._preview.load_source(next_source.file_path)
                    self._preview.seek_to_frame(next_clip.source_in, next_source.fps)
                    self._preview.play()
                    self._playback_clip = next_clip
                    self._playback_source = next_source
                    self._playback_clip_timeline_start = gap_end
                current_frame = min(current_frame, gap_end)

            self._playback_updating = True
            self._timeline_widget.set_playhead(current_frame)
            self._playback_updating = False
            self._timeline_widget.strip.ensure_playhead_visible()
            return

        # Normal clip playback: sync with mpv
        if not self._preview.is_playing:
            self._stop_playback()
            return

        time_pos = self._preview.get_time_pos()
        if time_pos < 0:
            return

        fps = self._playback_source.fps
        source_frame = int(time_pos * fps)

        # Check if we've passed the end of the current clip
        if source_frame > clip.source_out:
            # Move to next clip on the timeline
            next_start = self._playback_clip_timeline_start + clip.duration_frames
            result = self._timeline.get_clip_at_position(next_start)
            if result is None:
                self._stop_playback()
                return
            next_clip, _ = result
            if next_clip.is_gap:
                # Enter gap playback mode
                self._preview.pause()
                self._preview.show_black()
                self._playback_clip = next_clip
                self._playback_clip_timeline_start = next_start
                self._gap_start_time = time.monotonic()
                self._gap_start_frame = next_start
                return

            next_source = self._sources.get(next_clip.source_id)
            if not next_source:
                self._stop_playback()
                return

            # Check if next clip is contiguous in the same source — if so, just
            # update bookkeeping and let mpv keep playing without interruption
            contiguous = (
                next_source.file_path == self._playback_source.file_path
                and next_clip.source_in == clip.source_out + 1
            )

            self._playback_clip = next_clip
            self._playback_source = next_source
            self._playback_clip_timeline_start = next_start

            if not contiguous:
                self._preview.load_source(next_source.file_path)
                self._preview.seek_to_frame(next_clip.source_in, next_source.fps)
                self._preview.play()
            source_frame = max(source_frame, next_clip.source_in)

        # Update timeline playhead (rebind clip after possible boundary advance)
        clip = self._playback_clip
        offset_in_clip = source_frame - clip.source_in
        timeline_frame = self._playback_clip_timeline_start + offset_in_clip

        self._playback_updating = True
        self._timeline_widget.set_playhead(timeline_frame)
        self._playback_updating = False
        self._timeline_widget.strip.ensure_playhead_visible()
        self._status_frame.setText(f"Frame {timeline_frame}")

        if self._selection_follows_playhead:
            self._timeline.select_clip(self._playback_clip.id)

    # --- Export ---

    def _on_export_video(self):
        self._show_export_dialog(tab=0)

    def _on_export_images(self):
        self._show_export_dialog(tab=1)

    def _on_export_audio(self):
        self._show_export_dialog(tab=2)

    def _on_export_xml(self):
        # XML lives as a tab in the unified ExportDialog now.
        self._show_export_dialog(tab=3)

    def _run_xml_export(self, settings: dict, dialog):
        first_source = next(iter(self._sources.values()), None)
        fps = first_source.fps if first_source else 24.0
        export_fcpxml(
            self._timeline, self._sources, settings["output_path"],
            include_gaps=settings["include_gaps"],
            use_render_range=settings["use_render_range"],
            fps=fps,
            group_filter=settings.get("group_filter"),
        )

    def _on_export_otio(self):
        # OTIO lives as a tab in the unified ExportDialog now.
        self._show_export_dialog(tab=4)

    def _run_otio_export(self, settings: dict, dialog):
        first_source = next(iter(self._sources.values()), None)
        fps = first_source.fps if first_source else 24.0
        export_otio(
            self._timeline, self._sources, settings["output_path"],
            include_gaps=settings["include_gaps"],
            use_render_range=settings["use_render_range"],
            fps=fps,
            group_filter=settings.get("group_filter"),
        )

    def _show_export_dialog(self, tab: int = 0):
        if self._timeline.clip_count == 0:
            QMessageBox.information(self, "Export", "No clips to export.")
            return

        # Default resolution from first source
        first_source = next(iter(self._sources.values())) if self._sources else None
        w = first_source.width if first_source else 1920
        h = first_source.height if first_source else 1080
        fps = first_source.fps if first_source else 24.0
        # Calculate actual export frames (excluding gaps)
        export_frames = sum(c.duration_frames for c in self._timeline.clips if not c.is_gap)
        has_in_out = self._timeline.in_point is not None or self._timeline.out_point is not None
        render_frames = self._get_render_frame_count() if has_in_out else None
        clip_count = self._timeline.real_clip_count

        dialog = ExportDialog(w, h, fps, export_frames, render_frames=render_frames,
                              clip_count=clip_count, source_width=w, source_height=h,
                              timeline=self._timeline, parent=self)
        dialog._tabs.setCurrentIndex(tab)
        dialog.export_requested.connect(
            lambda settings: self._run_export(settings, dialog)
        )
        dialog.exec()

    def _run_export(self, settings: dict, dialog: ExportDialog):
        mode = settings.get("mode", "video")
        # XML and OTIO are timeline-interchange formats — single-shot export
        # via their respective writers, no Exporter / progress signals.
        if mode == "xml":
            self._run_xml_export(settings, dialog)
            dialog.export_finished()
            return
        if mode == "otio":
            self._run_otio_export(settings, dialog)
            dialog.export_finished()
            return
        self._exporter = Exporter(
            self._timeline, self._sources
        )
        self._exporter.progress.connect(dialog.set_progress, Qt.ConnectionType.QueuedConnection)
        self._exporter.status.connect(dialog.set_status, Qt.ConnectionType.QueuedConnection)
        self._exporter.finished.connect(dialog.export_finished, Qt.ConnectionType.QueuedConnection)
        self._exporter.cancelled.connect(dialog.export_cancelled, Qt.ConnectionType.QueuedConnection)
        self._exporter.error.connect(dialog.export_failed, Qt.ConnectionType.QueuedConnection)
        # Cancel button during an active export, and the X-close fallback.
        dialog.cancel_requested.connect(self._exporter.cancel)
        dialog.rejected.connect(self._exporter.cancel)
        self._exporter.export(settings)

    # --- Project management ---

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._autosave_timer.start()
            self._update_title()

    def _update_title(self):
        name = "Untitled"
        if self._project_path:
            name = os.path.basename(self._project_path)
        dirty = " *" if self._dirty else ""
        self.setWindowTitle(f"PrismaSynth v{__version__} — {name}{dirty}")

    def _on_new_project(self):
        if not self._confirm_discard():
            return
        self._stop_playback()

        if self._thumbnail_cache is not None:
            self._thumbnail_cache.stop()
            self._thumbnail_cache = None
        self._timeline.clear()
        self._reader_pool.close_all()
        self._proxy_manager.close_all()
        self._sources.clear()
        self._orphan_paths.clear()
        self._refresh_media_panel()
        self._project_path = None
        self._dirty = False
        self._preview.clear_frame()
        self._autosave_timer.stop()
        self._update_title()
        self._update_status()

    def _on_save_project(self):
        if self._project_path:
            self._save_to(self._project_path)
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "PrismaSynth Project (*.psynth)"
        )
        if path:
            if not path.endswith(".psynth"):
                path += ".psynth"
            self._save_to(path)

    def _save_to(self, path: str):
        try:
            playhead = self._timeline_widget.strip.playhead_frame
            scroll = self._timeline_widget.strip._scroll_offset
            zoom = self._timeline_widget.strip.pixels_per_frame
            save_project(path, self._sources, self._timeline.clips,
                         playhead, self._selection_follows_playhead,
                         in_point=self._timeline.in_point,
                         out_point=self._timeline.out_point,
                         scroll_offset=scroll,
                         pixels_per_frame=zoom,
                         orphan_paths=self._orphan_paths,
                         groups=self._timeline.groups,
                         top_splitter_sizes=self._top_splitter.sizes(),
                         vertical_splitter_sizes=self._main_splitter.sizes(),
                         right_tab_index=self._right_tabs.currentIndex())
            self._project_path = path
            self._dirty = False
            self._autosave_timer.stop()
            self._add_recent_file(path)
            self._update_title()
            logger.info("Project saved to %s", path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{e}")
            logger.exception("Save failed")

    def _on_open_project(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "PrismaSynth Project (*.psynth)"
        )
        if path:
            self._load_from(path)

    def _load_from(self, path: str):
        try:
            data = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open project:\n{e}")
            logger.exception("Load failed")
            return

        # Detect missing sources after the load_project relative-path fallback.
        # If the user cancels the relink dialog, abort BEFORE clearing state so
        # the previously-open project remains intact.
        relinks_applied = False
        missing = {sid: s for sid, s in data["sources"].items()
                   if not os.path.exists(s.file_path)}
        if missing:
            dlg = RelinkDialog(missing, parent=self)
            if dlg.exec() != QDialog.Accepted:
                return
            resolved = dlg.resolved_paths()
            probe_cache = dlg.probe_cache()
            shrunk_sources: set = set()
            for sid, new_path in resolved.items():
                if not new_path:
                    continue
                src = data["sources"][sid]
                info = probe_cache.get(sid)
                old_total_frames = src.total_frames
                src.file_path = new_path
                if info is not None:
                    src.width = info.width
                    src.height = info.height
                    src.fps = info.fps
                    src.total_frames = info.total_frames
                    src.codec = info.codec
                    src.audio_codec = info.audio_codec
                    src.audio_sample_rate = info.audio_sample_rate
                    src.audio_channels = info.audio_channels
                    src.time_base_num = info.time_base_num
                    src.time_base_den = info.time_base_den
                    if info.total_frames < old_total_frames:
                        shrunk_sources.add(sid)
                relinks_applied = True

            # Clamp any clip whose source_out exceeds the new (smaller) frame count.
            clamped = 0
            for c in data["clips"]:
                if c.source_id in shrunk_sources:
                    new_max = data["sources"][c.source_id].total_frames - 1
                    if new_max < 0:
                        continue
                    if c.source_in > new_max:
                        c.source_in = new_max
                        c.source_out = new_max
                        clamped += 1
                    elif c.source_out > new_max:
                        c.source_out = new_max
                        clamped += 1
            if clamped:
                QMessageBox.warning(
                    self, "Clips Clamped",
                    f"{clamped} clip(s) had their out-points trimmed because "
                    "the relinked source has fewer frames.",
                )

        # Clear current state (only after the relink dialog has accepted, so
        # cancelling the dialog leaves the previous project intact).
        self._stop_playback()

        self._timeline.clear()
        self._reader_pool.close_all()
        self._proxy_manager.close_all()
        self._sources.clear()
        self._preview.clear_frame()

        # Restore sources (update in-place to preserve references held by other objects)
        self._sources.clear()
        self._sources.update(data["sources"])
        for source in self._sources.values():
            # Skipped/broken sources have non-existent paths; don't spin up
            # readers or proxies for them. Clips referencing them will still
            # render (black preview, no thumbnails) — matches existing behaviour.
            if not os.path.exists(source.file_path):
                continue
            self._reader_pool.register_source(source)
            self._proxy_manager.load_or_open(source)
            # Kick off a background thumbnail extract for the Media Panel.
            # Cached thumbs return instantly; first-time projects extract once.
            self._thumb_pool.start(_SourceThumbWorker(source, self._thumb_extracted))

        # Populate the Media Panel
        self._refresh_media_panel()

        # Restore the orphan-paths registry so a re-import after reopen still
        # reattaches the orphan clips to a freshly-imported source.
        self._orphan_paths.clear()
        self._orphan_paths.update(data.get("orphan_paths", {}))

        # Set FPS from first source
        if self._sources:
            fps = next(iter(self._sources.values())).fps
            self._timeline_widget.set_fps(fps)

        # Restore groups BEFORE clips so the clips' group_ids reference a
        # populated registry by the time clip-related signals fire.
        self._timeline.set_groups_bulk(data.get("groups", {}).values())

        # Restore clips (preserve saved colors)
        self._timeline.add_clips(data["clips"], assign_colors=False)

        # Restore zoom BEFORE the scroll offset — scroll_offset is a pixel
        # value, so it only lands at the correct logical position once
        # _pixels_per_frame matches what was saved.
        self._timeline_widget.strip.set_pixels_per_frame(
            data.get("pixels_per_frame", 0.5))

        # Restore playhead and scroll position
        self._timeline_widget.set_playhead(data.get("playhead_position", 0))
        self._timeline_widget.strip.set_scroll_offset(data.get("scroll_offset", 0))

        # Restore in/out points
        in_pt = data.get("in_point")
        out_pt = data.get("out_point")
        if in_pt is not None:
            self._timeline.set_in_point(in_pt)
        if out_pt is not None:
            self._timeline.set_out_point(out_pt)

        # Surface the seek-drift warning for any source whose container
        # time_base can't exactly represent its declared fps. Fires once per
        # project load; users dismiss the dialog after acting on it.
        self._warn_unsafe_timebases(list(self._sources.values()))

        # Restore selection follows + sync toolbar toggle
        self._selection_follows_playhead = data.get("selection_follows_playhead", True)
        self._toolbar._selection_follows_action.setChecked(self._selection_follows_playhead)

        # Restore per-project workspace layout (project wins; absent on
        # legacy projects, in which case the QSettings/default layout
        # already in effect at startup is preserved untouched).
        saved_top = data.get("top_splitter_sizes")
        if saved_top:
            try:
                self._top_splitter.setSizes([int(s) for s in saved_top])
            except (TypeError, ValueError):
                pass

        saved_vert = data.get("vertical_splitter_sizes")
        if saved_vert:
            try:
                self._main_splitter.setSizes([int(s) for s in saved_vert])
            except (TypeError, ValueError):
                pass

        saved_tab = data.get("right_tab_index")
        if saved_tab is not None:
            try:
                idx = int(saved_tab)
                if 0 <= idx < self._right_tabs.count():
                    self._right_tabs.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass

        # Start thumbnails
        self._start_thumbnail_cache()

        # Update state
        self._project_path = path
        self._dirty = relinks_applied
        if relinks_applied:
            # Restart autosave so the new paths get persisted within 60s.
            self._autosave_timer.start()
        else:
            self._autosave_timer.stop()
        self._add_recent_file(path)
        self._update_title()
        self._update_status()
        logger.info("Project loaded from %s: %d sources, %d clips",
                     path, len(self._sources), self._timeline.clip_count)

    def _autosave(self):
        if self._dirty and self._project_path:
            self._save_to(self._project_path)

    def _confirm_discard(self) -> bool:
        """Ask user to save if there are unsaved changes. Returns True to proceed."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_save_project()
            return not self._dirty  # proceed only if save succeeded
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    # --- Recent files ---

    def _recent_files_path(self) -> str:
        return str(get_config_dir() / "recent_projects.json")

    def _load_recent_files(self):
        try:
            with open(self._recent_files_path(), "r") as f:
                self._recent_files = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._recent_files = []

    def _save_recent_files(self):
        try:
            with open(self._recent_files_path(), "w") as f:
                json.dump(self._recent_files, f)
        except Exception:
            pass

    def _add_recent_file(self, path: str):
        path = os.path.abspath(path)
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:10]
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        if not self._recent_files:
            no_recent = QAction("(No recent projects)", self)
            no_recent.setEnabled(False)
            self._recent_menu.addAction(no_recent)
            return
        for path in self._recent_files:
            name = os.path.basename(path)
            action = QAction(name, self)
            action.setToolTip(path)
            action.triggered.connect(lambda checked, p=path: self._open_recent(p))
            self._recent_menu.addAction(action)

    def _open_recent(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found",
                                f"Project file not found:\n{path}")
            self._recent_files = [p for p in self._recent_files if p != path]
            self._save_recent_files()
            self._rebuild_recent_menu()
            return
        if not self._confirm_discard():
            return
        self._load_from(path)

    # --- Status ---

    def _update_status(self):
        count = self._timeline.real_clip_count
        total_frames = self._timeline.get_total_duration_frames()
        fps = 24.0
        if self._sources:
            fps = next(iter(self._sources.values())).fps
        total_secs = total_frames / fps if fps > 0 else 0
        minutes = int(total_secs) // 60
        secs = int(total_secs) % 60
        self._status_clips.setText(f"{count} clips")
        self._status_duration.setText(f"{minutes:02d}:{secs:02d}")

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self._stop_playback()
        self._autosave_timer.stop()
        if self._thumbnail_cache:
            self._thumbnail_cache.stop()

        self._preview.cleanup()
        self._reader_pool.close_all()
        self._proxy_manager.close_all()
        super().closeEvent(event)
