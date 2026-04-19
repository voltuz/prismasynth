import json
import logging
import os
import time
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStatusBar, QMessageBox, QFileDialog, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from core.timeline import TimelineModel
from core.clip import Clip
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool
from core.edl_exporter import export_edl
from core.exporter import Exporter
from core.thumbnail_cache import ThumbnailCache
from core.proxy_cache import ProxyManager
from core.project import save_project, load_project
from utils.paths import get_config_dir
from ui.icon_loader import icon
from ui.preview_widget import PreviewWidget
from ui.timeline_widget import TimelineWidget, EditMode
from ui.toolbar import MainToolbar
from ui.clip_info_panel import ClipInfoPanel
from ui.import_dialog import ImportDialog
from ui.detect_dialog import DetectDialog
from ui.export_dialog import ExportDialog

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
}
QMenu::item {
    padding: 4px 24px;
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PrismaSynth")
        self.setMinimumSize(1024, 600)
        self.resize(1400, 800)
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

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Top area: clip info + preview
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._clip_info = ClipInfoPanel()
        top_layout.addWidget(self._clip_info)

        self._preview = PreviewWidget()
        top_layout.addWidget(self._preview, 1)

        splitter.addWidget(top_widget)

        # Bottom area: timeline (with left/right margins)
        self._timeline_widget = TimelineWidget(self._timeline)
        timeline_container = QWidget()
        tl_layout = QVBoxLayout(timeline_container)
        tl_layout.setContentsMargins(16, 0, 16, 0)
        tl_layout.addWidget(self._timeline_widget)
        splitter.addWidget(timeline_container)

        splitter.setSizes([400, 200])

        # Timeline signals
        self._timeline_widget.playhead_changed.connect(self._on_playhead_changed)
        self._timeline_widget.scrub_started.connect(self._preview.scrub_start)
        self._timeline_widget.scrub_ended.connect(self._preview.scrub_end)
        self._timeline_widget.clip_clicked.connect(self._on_clip_clicked)
        self._timeline_widget.preview_frame_requested.connect(self._on_preview_frame_requested)
        self._timeline_widget.cut_requested.connect(self._on_cut_at_frame)
        self._timeline_widget.thumbnails_toggled.connect(self._on_thumbnails_toggled)
        self._timeline_widget.hq_thumbnails_toggled.connect(self._on_hq_thumbnails_toggled)
        self._timeline_widget.strip.scroll_changed.connect(self._on_viewport_changed)
        self._timeline.selection_changed.connect(self._on_selection_changed)

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

    def showEvent(self, event):
        super().showEvent(event)
        # Init mpv after widget is shown (needs valid winId)
        self._preview.init_player()

    def _setup_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        new_action = QAction(icon("new"), "New Project", self)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction(icon("open"), "Open Project...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        save_action = QAction(icon("save"), "Save Project", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction(icon("save-as"), "Save Project As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        edit_menu = menu.addMenu("Edit")

        split_action = QAction(icon("scissors"), "Split at Playhead", self)
        split_action.setShortcut("S")  # middle finger home
        split_action.triggered.connect(self._on_split)
        edit_menu.addAction(split_action)

        delete_action = QAction(icon("trash"), "Delete Selected", self)
        delete_action.setShortcut("W")  # top row middle — most-used delete
        delete_action.triggered.connect(self._on_delete)
        edit_menu.addAction(delete_action)

        ripple_delete_action = QAction(icon("ripple-delete"), "Ripple Delete Selected", self)
        ripple_delete_action.setShortcut("D")  # index finger home — ripple delete
        ripple_delete_action.triggered.connect(self._on_ripple_delete)
        edit_menu.addAction(ripple_delete_action)

        edit_menu.addSeparator()

        undo_action = QAction(icon("undo"), "Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self._timeline.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction(icon("redo"), "Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_action.triggered.connect(self._timeline.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction(icon("select-all"), "Select All", self)
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))
        select_all_action.triggered.connect(self._timeline.select_all)
        edit_menu.addAction(select_all_action)

        select_gap_action = QAction(icon("select-gap"), "Select to Gap Left", self)
        select_gap_action.setShortcut("A")  # pinky home — select to gap
        select_gap_action.triggered.connect(self._on_select_to_gap)
        edit_menu.addAction(select_gap_action)

        edit_menu.addSeparator()

        set_in_action = QAction(icon("set-in"), "Set In Point", self)
        set_in_action.setShortcut("E")  # top row — in point
        set_in_action.triggered.connect(self._on_set_in_point)
        edit_menu.addAction(set_in_action)

        set_out_action = QAction(icon("set-out"), "Set Out Point", self)
        set_out_action.setShortcut("R")  # top row — out point
        set_out_action.triggered.connect(self._on_set_out_point)
        edit_menu.addAction(set_out_action)

        clear_in_out_action = QAction(icon("clear-in-out"), "Clear In/Out", self)
        clear_in_out_action.setShortcut("X")
        clear_in_out_action.triggered.connect(self._on_clear_in_out)
        edit_menu.addAction(clear_in_out_action)

        edit_menu.addSeparator()

        selection_mode_action = QAction(icon("cursor"), "Selection Mode", self)
        selection_mode_action.setShortcut("V")
        selection_mode_action.triggered.connect(lambda: self._set_edit_mode(EditMode.SELECTION))
        edit_menu.addAction(selection_mode_action)

        cut_mode_action = QAction(icon("cut-mode"), "Cut Mode", self)
        cut_mode_action.setShortcut("C")
        cut_mode_action.triggered.connect(lambda: self._set_edit_mode(EditMode.CUT))
        edit_menu.addAction(cut_mode_action)

        scrub_follow_action = QAction(icon("target"), "Scrub Follow", self)
        scrub_follow_action.setShortcut("F")
        scrub_follow_action.triggered.connect(self._toggle_scrub_follow)
        edit_menu.addAction(scrub_follow_action)

        # --- Timeline menu ---
        timeline_menu = menu.addMenu("Timeline")

        import_action = QAction(icon("import"), "Import Video...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self._on_import)
        timeline_menu.addAction(import_action)

        timeline_menu.addSeparator()

        export_vid_action = QAction(icon("film"), "Export Video...", self)
        export_vid_action.setShortcut(QKeySequence("Ctrl+E"))
        export_vid_action.triggered.connect(self._on_export_video)
        timeline_menu.addAction(export_vid_action)

        export_img_action = QAction(icon("image"), "Export Image Sequence...", self)
        export_img_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_img_action.triggered.connect(self._on_export_images)
        timeline_menu.addAction(export_img_action)

        edl_action = QAction(icon("document"), "Export EDL...", self)
        edl_action.triggered.connect(self._on_export_edl)
        timeline_menu.addAction(edl_action)

        timeline_menu.addSeparator()

        detect_action = QAction(icon("wand"), "Detect Cuts...", self)
        detect_action.setShortcut(QKeySequence("Ctrl+D"))
        detect_action.triggered.connect(self._on_detect_cuts)
        timeline_menu.addAction(detect_action)

        timeline_menu.addSeparator()

        play_action = QAction(icon("play"), "Play/Pause", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self._toggle_play)
        timeline_menu.addAction(play_action)

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
        """Import video files directly (used by drag-and-drop).
        Validates resolution/FPS against existing timeline sources."""
        from utils.ffprobe import probe_video
        ref_w, ref_h, ref_fps = self._get_timeline_ref()

        probed = []
        for path in file_paths:
            info = probe_video(path)
            if info is None:
                logger.warning("Could not probe dropped file: %s", path)
                return
            probed.append((path, info))

        # Use first file as reference if timeline is empty
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

        results = []
        for path, info in probed:
            source = VideoSource(
                file_path=path,
                total_frames=info.total_frames,
                fps=info.fps,
                width=info.width,
                height=info.height,
                codec=info.codec,
            )
            clip = Clip(source_id=source.id, source_in=0,
                        source_out=source.total_frames - 1)
            results.append((source, clip))

        self._on_import_complete(results)

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

    def _on_import_complete(self, results: list):
        """Handle import of one or more (VideoSource, Clip) pairs."""
        if not results:
            return
        all_clips = []
        for source, clip in results:
            self._sources[source.id] = source
            self._reader_pool.register_source(source)
            self._proxy_manager.load_or_open(source)
            all_clips.append(clip)

        self._timeline.add_clips(all_clips)
        self._timeline_widget.set_fps(results[0][0].fps)

        # Start thumbnail generation
        self._start_thumbnail_cache()

        # Show first frame
        if all_clips:
            self._timeline_widget.set_playhead(0)

        self._update_status()
        for source, clip in results:
            logger.info("Imported %s", source.file_path)

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
            self._thumbnail_cache.thumbnail_ready.connect(self._on_thumbnail_ready)
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

        # Pause thumbnail generation while scrubbing — resume after 500ms idle
        if self._thumbnail_cache:
            self._thumbnail_cache.pause()
            self._thumb_resume_timer.start()

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
        if self._thumbnail_cache:
            self._thumbnail_cache.pause()
            self._thumb_resume_timer.start()

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
        if self._preview.is_playing:
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

    def _on_export_edl(self):
        if self._timeline.clip_count == 0:
            QMessageBox.information(self, "Export EDL", "No clips to export.")
            return
        from ui.edl_dialog import EdlDialog
        clip_count = self._timeline.real_clip_count
        total_frames = sum(c.duration_frames for c in self._timeline.clips
                           if not c.is_gap)
        first_source = next(iter(self._sources.values()), None)
        fps = first_source.fps if first_source else 24.0
        has_range = (self._timeline.in_point is not None
                     or self._timeline.out_point is not None)
        dialog = EdlDialog(clip_count, total_frames, fps,
                           has_render_range=has_range, parent=self)
        dialog.export_requested.connect(
            lambda s: self._run_edl_export(s, dialog))
        dialog.exec()

    def _run_edl_export(self, settings: dict, dialog):
        first_source = next(iter(self._sources.values()), None)
        fps = first_source.fps if first_source else 24.0
        export_edl(
            self._timeline, self._sources, settings["output_path"],
            include_gaps=settings["include_gaps"],
            use_render_range=settings["use_render_range"],
            fps=fps,
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
                              parent=self)
        dialog._tabs.setCurrentIndex(tab)
        dialog.export_requested.connect(
            lambda settings: self._run_export(settings, dialog)
        )
        dialog.exec()

    def _run_export(self, settings: dict, dialog: ExportDialog):
        self._exporter = Exporter(
            self._timeline, self._sources
        )
        self._exporter.progress.connect(dialog.set_progress, Qt.ConnectionType.QueuedConnection)
        self._exporter.status.connect(dialog.set_status, Qt.ConnectionType.QueuedConnection)
        self._exporter.finished.connect(dialog.export_finished, Qt.ConnectionType.QueuedConnection)
        self._exporter.error.connect(lambda msg: dialog.set_status(f"Error: {msg}"),
                                     Qt.ConnectionType.QueuedConnection)
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
        self.setWindowTitle(f"PrismaSynth — {name}{dirty}")

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
            save_project(path, self._sources, self._timeline.clips,
                         playhead, self._selection_follows_playhead,
                         in_point=self._timeline.in_point,
                         out_point=self._timeline.out_point,
                         scroll_offset=scroll)
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

        # Clear current state
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
            self._reader_pool.register_source(source)
            self._proxy_manager.load_or_open(source)

        # Set FPS from first source
        if self._sources:
            fps = next(iter(self._sources.values())).fps
            self._timeline_widget.set_fps(fps)

        # Restore clips (preserve saved colors)
        self._timeline.add_clips(data["clips"], assign_colors=False)

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

        # Restore selection follows + sync toolbar toggle
        self._selection_follows_playhead = data.get("selection_follows_playhead", True)
        self._toolbar._selection_follows_action.setChecked(self._selection_follows_playhead)

        # Start thumbnails
        self._start_thumbnail_cache()

        # Update state
        self._project_path = path
        self._dirty = False
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
