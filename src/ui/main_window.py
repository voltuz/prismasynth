import json
import logging
import os
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStatusBar, QMessageBox, QFileDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from core.timeline import TimelineModel
from core.clip import Clip
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool
from core.exporter import Exporter
from core.thumbnail_cache import ThumbnailCache
from core.proxy_cache import ProxyManager, ProxyFile, HQProxyGenerator
from core.project import save_project, load_project
from utils.paths import get_config_dir
from ui.preview_widget import PreviewWidget
from ui.timeline_widget import TimelineWidget
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
        self._hq_proxy_gen = HQProxyGenerator()
        self._hq_proxy_gen.finished.connect(self._on_hq_proxy_ready)
        self._exporter: Optional[Exporter] = None
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

        # Playback — mpv plays natively, timer syncs the timeline playhead
        self._playback_timer = QTimer()
        self._playback_timer.setInterval(16)  # ~60Hz playhead sync
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_source = None
        self._playback_clip = None
        self._playback_clip_timeline_start = 0

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
        self._toolbar.export_video_clicked.connect(self._on_export_video)
        self._toolbar.export_images_clicked.connect(self._on_export_images)

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

        # Bottom area: timeline
        self._timeline_widget = TimelineWidget(self._timeline)
        splitter.addWidget(self._timeline_widget)

        splitter.setSizes([400, 200])

        # Timeline signals
        self._timeline_widget.playhead_changed.connect(self._on_playhead_changed)
        self._timeline_widget.clip_clicked.connect(self._on_clip_clicked)
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

        new_action = QAction("New Project", self)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("Open Project...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        save_action = QAction("Save Project", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Project As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        import_action = QAction("Import Video...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        export_vid_action = QAction("Export Video...", self)
        export_vid_action.setShortcut(QKeySequence("Ctrl+E"))
        export_vid_action.triggered.connect(self._on_export_video)
        file_menu.addAction(export_vid_action)

        export_img_action = QAction("Export Image Sequence...", self)
        export_img_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_img_action.triggered.connect(self._on_export_images)
        file_menu.addAction(export_img_action)

        detect_action = QAction("Detect Cuts...", self)
        detect_action.setShortcut(QKeySequence("Ctrl+D"))
        detect_action.triggered.connect(self._on_detect_cuts)
        file_menu.addAction(detect_action)

        play_action = QAction("Play/Pause", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self._toggle_play)
        file_menu.addAction(play_action)

        edit_menu = menu.addMenu("Edit")

        split_action = QAction("Split at Playhead", self)
        split_action.setShortcut("S")
        split_action.triggered.connect(self._on_split)
        edit_menu.addAction(split_action)

        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut("Backspace")
        delete_action.triggered.connect(self._on_delete)
        edit_menu.addAction(delete_action)

        ripple_delete_action = QAction("Ripple Delete Selected", self)
        ripple_delete_action.setShortcut("Delete")
        ripple_delete_action.triggered.connect(self._on_ripple_delete)
        edit_menu.addAction(ripple_delete_action)

        edit_menu.addSeparator()

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self._timeline.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_action.triggered.connect(self._timeline.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))
        select_all_action.triggered.connect(self._timeline.select_all)
        edit_menu.addAction(select_all_action)

        select_gap_action = QAction("Select to Gap Left", self)
        select_gap_action.setShortcut("G")
        select_gap_action.triggered.connect(self._on_select_to_gap)
        edit_menu.addAction(select_gap_action)

    # --- Import ---

    _VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mxf'}

    def _on_import(self):
        dialog = ImportDialog(self)
        dialog.import_complete.connect(self._on_import_complete)
        dialog.exec()

    def _import_file(self, file_path: str):
        """Import a video file directly (used by drag-and-drop)."""
        from utils.ffprobe import probe_video
        info = probe_video(file_path)
        if info is None:
            logger.warning("Could not probe dropped file: %s", file_path)
            return
        source = VideoSource(
            file_path=file_path,
            total_frames=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            codec=info.codec,
        )
        clip = Clip(source_id=source.id, source_in=0, source_out=source.total_frames - 1)
        self._on_import_complete(source, [clip])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in self._VIDEO_EXTENSIONS:
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in self._VIDEO_EXTENSIONS:
                    self._import_file(path)

    def _on_import_complete(self, source: VideoSource, clips):
        self._sources[source.id] = source
        self._reader_pool.register_source(source)
        self._timeline.add_clips(clips)
        self._timeline_widget.set_fps(source.fps)

        # Load any existing proxy
        self._proxy_manager.load_or_open(source)

        # Start background HQ proxy generation (960x540 JPEG — doesn't block)
        self._hq_proxy_gen.generate(source)

        # Start thumbnail generation
        self._start_thumbnail_cache()

        # Show first frame
        if clips:
            self._timeline_widget.set_playhead(0)

        self._update_status()
        logger.info("Imported %s: %d clips", source.file_path, len(clips))

    def _on_detect_cuts(self):
        """Run cut detection on a source. Uses the first source if only one,
        or the source of the selected clip."""
        if not self._sources:
            return
        # Pick source: from selected clip, or first available
        source = None
        selected = self._timeline.get_selected_clips()
        if selected and selected[0].source_id:
            source = self._sources.get(selected[0].source_id)
        if source is None:
            source = next(iter(self._sources.values()))

        dialog = DetectDialog(source, self)
        dialog.detection_complete.connect(self._on_detection_complete)
        dialog.exec()

    def _on_detection_complete(self, source_id: str, clips):
        """Replace all clips from this source with the detected scene clips."""
        # Remove existing clips for this source entirely (not gap-replace)
        self._timeline.ripple_delete_by_source(source_id)
        # Add new scene clips
        self._timeline.add_clips(clips)

        # Load proxy saved by scene detector
        source = self._sources.get(source_id)
        if source:
            self._proxy_manager.load_or_open(source)
            self._hq_proxy_gen.generate(source)

        self._start_thumbnail_cache()
        if clips:
            self._timeline_widget.set_playhead(0)
        self._update_status()
        logger.info("Detected %d cuts for source %s", len(clips), source_id)

    def _start_thumbnail_cache(self):
        if self._thumbnail_cache is not None:
            self._thumbnail_cache.stop()
        self._thumbnail_cache = ThumbnailCache(
            self._reader_pool, self._timeline, self._sources,
            proxy_manager=self._proxy_manager,
        )
        self._thumbnail_cache.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumbnail_cache.generate_all()

    def _on_thumbnail_ready(self, clip_id: str, position: str, qimage):
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap.fromImage(qimage)
        self._timeline_widget.strip.set_thumbnail(clip_id, position, pixmap)

    def _on_hq_proxy_ready(self, source_id: str):
        """Background HQ proxy generation finished — upgrade the proxy."""
        source = self._sources.get(source_id)
        if source is not None:
            self._proxy_manager.upgrade_to_hq(source)
            logger.info("HQ proxy ready for %s", source.file_path)

    # --- Playhead ---

    def _on_playhead_changed(self, frame: int):
        # Playback-driven update — just sync UI, don't touch mpv
        if self._playback_updating:
            self._status_frame.setText(f"Frame {frame}")
            return

        # User moved the playhead — restart playback from new position if playing
        if self._preview.is_playing:
            self._stop_playback()
            self._timeline_widget.strip._playhead_frame = frame
            self._start_playback()
            return

        self._status_frame.setText(f"Frame {frame}")

        if self._selection_follows_playhead:
            result = self._timeline.get_clip_at_position(frame)
            if result:
                clip, _ = result
                self._timeline.select_clip(clip.id)

        # Pause thumbnail generation while scrubbing — resume after 500ms idle
        if self._thumbnail_cache:
            self._thumbnail_cache.pause()
            self._thumb_resume_timer.start()

        # Seek mpv to the correct source frame (GPU-accelerated)
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

    def _on_delete(self):
        """Delete selected clips, replacing with gaps (Backspace)."""
        if not self._timeline.selected_ids:
            return
        self._timeline.delete_selected()
        self._refresh_preview_after_edit()

    def _on_ripple_delete(self):
        """Ripple delete selected clips — collapse the space (Delete)."""
        if not self._timeline.selected_ids:
            return
        self._timeline.ripple_delete_selected()
        self._refresh_preview_after_edit()

    def _refresh_preview_after_edit(self):
        """Refresh preview after a timeline edit (delete, ripple delete, etc.)."""
        frame = self._timeline_widget.strip.playhead_frame
        self._on_playhead_changed(frame)

    def _on_select_to_gap(self):
        frame = self._timeline_widget.strip.playhead_frame
        self._timeline.select_to_gap_left(frame)

    # --- Playback ---

    def _toggle_play(self):
        if self._preview.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self._timeline.clip_count == 0:
            return
        # Ensure mpv has the right source loaded at the playhead position
        frame = self._timeline_widget.strip.playhead_frame
        result = self._timeline.get_clip_at_position(frame)
        if result is None or result[0].is_gap:
            return
        clip, offset = result
        source = self._sources.get(clip.source_id)
        if not source:
            return

        self._preview.load_source(source.file_path)
        self._preview.seek_to_frame(clip.source_in + offset, source.fps)
        self._preview.play()
        self._toolbar.set_playing(True)

        # Track mpv's playback position and update the timeline playhead
        self._playback_source = source
        self._playback_clip = clip
        self._playback_clip_timeline_start = self._timeline.get_clip_timeline_start(clip.id)
        self._playback_timer.start()

    def _stop_playback(self):
        self._playback_timer.stop()
        self._preview.pause()
        self._toolbar.set_playing(False)

    def _on_playback_tick(self):
        """Sync timeline playhead with mpv's current position during native playback."""
        if not self._preview.is_playing:
            self._stop_playback()
            return

        time_pos = self._preview.get_time_pos()
        if time_pos < 0:
            return

        fps = self._playback_source.fps
        source_frame = int(time_pos * fps)

        # Check if we've passed the end of the current clip
        clip = self._playback_clip
        if source_frame > clip.source_out:
            # Move to next clip on the timeline
            next_start = self._playback_clip_timeline_start + clip.duration_frames
            result = self._timeline.get_clip_at_position(next_start)
            if result is None:
                self._stop_playback()
                return
            next_clip, _ = result
            if next_clip.is_gap:
                # Skip gap, find next real clip
                gap_end = next_start + next_clip.duration_frames
                result = self._timeline.get_clip_at_position(gap_end)
                if result is None or result[0].is_gap:
                    self._stop_playback()
                    return
                next_clip, _ = result
                next_start = gap_end

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

    def _show_export_dialog(self, tab: int = 0):
        if self._timeline.clip_count == 0:
            QMessageBox.information(self, "Export", "No clips to export.")
            return

        # Default resolution from first source
        first_source = next(iter(self._sources.values())) if self._sources else None
        w = first_source.width if first_source else 1920
        h = first_source.height if first_source else 1080
        fps = first_source.fps if first_source else 24.0
        total = self._timeline.get_total_duration_frames()

        dialog = ExportDialog(w, h, fps, total, self)
        dialog._tabs.setCurrentIndex(tab)
        dialog.export_requested.connect(
            lambda settings: self._run_export(settings, dialog)
        )
        dialog.exec()

    def _run_export(self, settings: dict, dialog: ExportDialog):
        self._exporter = Exporter(
            self._timeline, self._sources, self._reader_pool
        )
        self._exporter.progress.connect(dialog.set_progress)
        self._exporter.status.connect(dialog.set_status)
        self._exporter.finished.connect(dialog.export_finished)
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
        self._hq_proxy_gen.stop()
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
            save_project(path, self._sources, self._timeline.clips,
                         playhead, self._selection_follows_playhead)
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
        self._hq_proxy_gen.stop()
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
            # Generate HQ proxy if only legacy proxy exists
            self._hq_proxy_gen.generate(source)

        # Set FPS from first source
        if self._sources:
            fps = next(iter(self._sources.values())).fps
            self._timeline_widget.set_fps(fps)

        # Restore clips (preserve saved colors)
        self._timeline.add_clips(data["clips"], assign_colors=False)

        # Restore playhead
        self._timeline_widget.set_playhead(data.get("playhead_position", 0))

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
        self._hq_proxy_gen.stop()
        self._preview.cleanup()
        self._reader_pool.close_all()
        self._proxy_manager.close_all()
        super().closeEvent(event)
