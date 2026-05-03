"""Media Panel — the imported-source list, similar to DaVinci Resolve's
Media Pool. Shows one entry per VideoSource (not per timeline clip).

Two view modes selectable via header icons:
  - Grid: large thumbnail with filename underneath (default).
  - List: small thumbnail with filename on a single row.

The user can:
  - Drag video files INTO the panel to add new sources to the pool.
  - Drag selected sources OUT of the panel onto the timeline to insert
    new clips at the drop position (handled by TimelineWidget on receive).
  - Double-click a source to open its SourceInfoDialog.
  - Right-click a source for a context menu (Open Info, Relink, Remove).

Custom MIME type for the pool→timeline drag:
  ``application/x-prismasynth-source-ids`` whose payload is newline-separated
  source IDs. The timeline widget watches for this MIME on dragEnterEvent.
"""

import os
from pathlib import Path
from typing import Dict

from PySide6.QtCore import (
    QByteArray, QMimeData, QSize, Qt, QSettings, Signal,
)
from PySide6.QtGui import (
    QAction, QDrag, QIcon, QPixmap, QStandardItem, QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListView, QMenu, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from core.video_source import VideoSource
from core.source_thumbnail import cache_path_for, THUMB_HEIGHT, THUMB_WIDTH
from ui.icon_loader import icon


SOURCE_ID_MIME = "application/x-prismasynth-source-ids"
SOURCE_DURATIONS_MIME = "application/x-prismasynth-source-durations"
_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv',
                     '.flv', '.webm', '.m4v', '.ts', '.mxf'}

# Custom item-data role used by the drag mime to look up each source's
# duration. Qt.ItemDataRole.UserRole holds the source ID; UserRole+1 holds
# the total_frames for the drag-preview footprint on the timeline.
_DURATION_ROLE = Qt.ItemDataRole.UserRole + 1


class _SourceModel(QStandardItemModel):
    """Item model that exposes our custom MIME on outgoing drags.

    Overriding ``mimeTypes`` + ``mimeData`` lets Qt's built-in QAbstractItemView
    drag detection handle the press-on-item → drag-out flow correctly. The
    earlier approach (overriding QListView.startDrag) was bypassed when the
    user pressed in any IconMode empty area, leaving only marquee selection.
    """

    def mimeTypes(self):
        return [SOURCE_ID_MIME, SOURCE_DURATIONS_MIME]

    def mimeData(self, indexes):
        ids = []
        durations = []
        seen = set()
        for idx in indexes:
            if not idx.isValid():
                continue
            sid = self.data(idx, Qt.ItemDataRole.UserRole)
            dur = self.data(idx, _DURATION_ROLE) or 0
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
                durations.append(int(dur))
        if not ids:
            return None
        mime = QMimeData()
        mime.setData(SOURCE_ID_MIME,
                     QByteArray("\n".join(ids).encode("utf-8")))
        mime.setData(SOURCE_DURATIONS_MIME,
                     QByteArray("\n".join(str(d) for d in durations).encode("utf-8")))
        return mime


class _SourceListView(QListView):
    """List view that accepts file-URL drops for adding new sources. Outgoing
    drags are produced by the model's mimeData() override above."""

    files_dropped = Signal(list)  # list[str] of local file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False)  # we don't reorder; only accept files
        self.setDragDropMode(QListView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setUniformItemSizes(True)

        # State for the manual drag override below.
        self._press_pos = None
        self._press_idx = None

    # --- Outgoing drag: manual press/move detection ---
    #
    # In IconMode + ExtendedSelection + Movement.Static, Qt's built-in drag
    # detection prefers rubber-band selection over starting a drag (even
    # when pressing on a thumbnail). Forcing a drag on item-press here
    # bypasses that heuristic entirely. Empty-area press still falls
    # through to super() and produces the marquee selection.

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            idx = self.indexAt(self._press_pos)
            self._press_idx = idx if idx.isValid() else None
        else:
            self._press_pos = None
            self._press_idx = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._press_idx is not None
                and self._press_pos is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            distance = (event.position().toPoint() - self._press_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                indexes = self.selectedIndexes() or [self._press_idx]
                mime = self.model().mimeData(indexes)
                if mime is not None:
                    drag = QDrag(self)
                    drag.setMimeData(mime)
                    # Drag pixmap = first item's icon, scaled. Improves the
                    # visual feedback while dragging vs. the default cursor.
                    ic = self.model().data(indexes[0], Qt.ItemDataRole.DecorationRole)
                    if isinstance(ic, QIcon) and not ic.isNull():
                        pix = ic.pixmap(THUMB_WIDTH, THUMB_HEIGHT)
                        drag.setPixmap(pix)
                        drag.setHotSpot(pix.rect().center())
                    drag.exec(Qt.DropAction.CopyAction)
                self._press_idx = None
                self._press_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_idx = None
        self._press_pos = None
        super().mouseReleaseEvent(event)

    # --- Incoming drag (file URLs to add to pool) ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self._urls_have_video(event.mimeData().urls()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()
                     if u.isLocalFile()
                     and os.path.splitext(u.toLocalFile())[1].lower() in _VIDEO_EXTENSIONS]
            if paths:
                self.files_dropped.emit(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    @staticmethod
    def _urls_have_video(urls) -> bool:
        for u in urls:
            if u.isLocalFile():
                ext = os.path.splitext(u.toLocalFile())[1].lower()
                if ext in _VIDEO_EXTENSIONS:
                    return True
        return False


class MediaPanel(QWidget):
    """Media Pool — the list of imported source videos."""

    # Signals consumed by MainWindow
    source_double_clicked = Signal(str)   # source_id
    relink_requested = Signal(str)        # source_id
    remove_requested = Signal(str)        # source_id
    files_dropped = Signal(list)          # list[str] — added to pool

    _SETTINGS_VIEW_MODE = "media_panel/view_mode"

    def __init__(self, parent=None):
        super().__init__(parent)
        # Resizable in a horizontal QSplitter; Preferred lets the user drag
        # the splitter handle, with a sensible floor so the panel never
        # collapses below readable.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Header (title + view-mode toggles) ---
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Media")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ddd;")
        header.addWidget(title)
        header.addStretch()

        # Grid / List toggles — distinct SVG icons (4-square grid vs.
        # 3-line list with leading bullets).
        self._grid_btn = QToolButton()
        self._grid_btn.setCheckable(True)
        self._grid_btn.setIcon(icon("grid"))
        self._grid_btn.setIconSize(QSize(16, 16))
        self._grid_btn.setToolTip("Grid view")
        self._grid_btn.setAutoRaise(True)
        self._grid_btn.clicked.connect(lambda: self._set_view_mode("grid"))
        header.addWidget(self._grid_btn)

        self._list_btn = QToolButton()
        self._list_btn.setCheckable(True)
        self._list_btn.setIcon(icon("list"))
        self._list_btn.setIconSize(QSize(16, 16))
        self._list_btn.setToolTip("List view")
        self._list_btn.setAutoRaise(True)
        self._list_btn.clicked.connect(lambda: self._set_view_mode("list"))
        header.addWidget(self._list_btn)

        layout.addLayout(header)

        # --- List view ---
        self._view = _SourceListView()
        self._model = _SourceModel()
        self._view.setModel(self._model)
        layout.addWidget(self._view, 1)

        # Empty-state hint label, shown over the (empty) list
        self._empty_label = QLabel(
            "No sources yet.\n\nDrag video files here\nor use File → Import."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(self._empty_label)

        # Wire view signals
        self._view.doubleClicked.connect(self._on_double_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_context_menu)
        self._view.files_dropped.connect(self.files_dropped)

        # Restore last-used view mode (defaults to grid)
        settings = QSettings()
        mode = settings.value(self._SETTINGS_VIEW_MODE, "grid", type=str)
        self._set_view_mode(mode if mode in ("grid", "list") else "grid")

    # --- Public API ---

    def set_sources(self, sources: Dict[str, VideoSource]):
        """Replace the entire pool contents. Re-renders all icons.
        Called whenever the source dict changes (import, remove, project load)."""
        self._model.clear()
        for sid, src in sources.items():
            item = QStandardItem(self._display_name(src))
            item.setData(sid, Qt.ItemDataRole.UserRole)
            item.setData(int(src.total_frames), _DURATION_ROLE)
            item.setEditable(False)
            item.setIcon(self._load_icon(src))
            item.setToolTip(self._tooltip(src))
            self._model.appendRow(item)
        self._refresh_empty_state()

    def refresh_thumbnail(self, source: VideoSource):
        """Force-reload the icon for one source (e.g. after async thumb extract)."""
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == source.id:
                item.setIcon(self._load_icon(source))
                return

    # --- View mode ---

    def _set_view_mode(self, mode: str):
        if mode == "grid":
            self._view.setViewMode(QListView.ViewMode.IconMode)
            self._view.setIconSize(QSize(THUMB_WIDTH, THUMB_HEIGHT))
            self._view.setGridSize(QSize(THUMB_WIDTH + 12, THUMB_HEIGHT + 28))
            self._view.setWordWrap(True)
            self._view.setResizeMode(QListView.ResizeMode.Adjust)
            self._view.setMovement(QListView.Movement.Static)
            self._view.setFlow(QListView.Flow.LeftToRight)
            self._view.setSpacing(4)
        else:  # list
            self._view.setViewMode(QListView.ViewMode.ListMode)
            self._view.setIconSize(QSize(48, 27))
            self._view.setGridSize(QSize())  # use default per-row sizing
            self._view.setWordWrap(False)
            self._view.setResizeMode(QListView.ResizeMode.Fixed)
            self._view.setMovement(QListView.Movement.Static)
            self._view.setFlow(QListView.Flow.TopToBottom)
            self._view.setSpacing(0)

        # Force a full re-layout. Without this, Qt sometimes carries stale
        # per-item positions (from a previous IconMode session) into ListMode,
        # which can hide rows below the viewport.
        self._view.scheduleDelayedItemsLayout()

        self._grid_btn.setChecked(mode == "grid")
        self._list_btn.setChecked(mode == "list")
        QSettings().setValue(self._SETTINGS_VIEW_MODE, mode)

    # --- Slots ---

    def _on_double_clicked(self, index):
        sid = self._model.data(index, Qt.ItemDataRole.UserRole)
        if sid:
            self.source_double_clicked.emit(sid)

    def _on_context_menu(self, pos):
        idx = self._view.indexAt(pos)
        if not idx.isValid():
            return
        sid = self._model.data(idx, Qt.ItemDataRole.UserRole)
        if not sid:
            return

        menu = QMenu(self)
        info_action = QAction("Open Info", menu)
        info_action.triggered.connect(lambda: self.source_double_clicked.emit(sid))
        menu.addAction(info_action)

        relink_action = QAction("Relink...", menu)
        relink_action.triggered.connect(lambda: self.relink_requested.emit(sid))
        menu.addAction(relink_action)

        menu.addSeparator()

        remove_action = QAction("Remove from Media Pool...", menu)
        remove_action.triggered.connect(lambda: self.remove_requested.emit(sid))
        menu.addAction(remove_action)

        menu.exec(self._view.viewport().mapToGlobal(pos))

    # --- Helpers ---

    def _refresh_empty_state(self):
        empty = (self._model.rowCount() == 0)
        self._view.setVisible(not empty)
        self._empty_label.setVisible(empty)

    @staticmethod
    def _display_name(src: VideoSource) -> str:
        return Path(src.file_path).stem or Path(src.file_path).name or src.id

    @staticmethod
    def _tooltip(src: VideoSource) -> str:
        return (f"{Path(src.file_path).name}\n"
                f"{src.width}×{src.height} • {src.fps:.3f} fps\n"
                f"Audio: {src.format_audio()}")

    @staticmethod
    def _load_icon(src: VideoSource) -> QIcon:
        path = cache_path_for(src)
        if path.exists() and path.stat().st_size > 0:
            pix = QPixmap(str(path))
            if not pix.isNull():
                return QIcon(pix)
        # Placeholder: solid-fill pixmap with a film-strip glyph would be nice
        # but for now an empty icon keeps the grid layout consistent.
        pix = QPixmap(THUMB_WIDTH, THUMB_HEIGHT)
        pix.fill(Qt.GlobalColor.darkGray)
        return QIcon(pix)
