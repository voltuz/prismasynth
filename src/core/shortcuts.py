"""Customizable keyboard-shortcut registry.

Single source of truth for every user-rebindable shortcut in the app. Each
entry has a stable `id` (referenced from MainWindow when wiring QActions or
QShortcuts), a category (the dialog's group header), a display `name`, and a
`default` key sequence string.

`ShortcutManager` owns the live state: it loads any saved overrides from
`QSettings`, hands out the current key to UI bindings on attach, and applies
reassignments back to those bindings without requiring a restart. The
manager enforces a uniqueness invariant — no two shortcuts may share the
same key sequence at the same time. This is by design, per the user's
request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut


@dataclass(frozen=True)
class ShortcutDef:
    id: str
    category: str
    name: str
    default: str  # QKeySequence portable string


# The full list of editable shortcuts, in the display order the dialog uses.
# Adding a new entry here automatically picks it up everywhere downstream.
SHORTCUTS: List[ShortcutDef] = [
    # File
    ShortcutDef("new_project",        "File",     "New Project",                "Ctrl+N"),
    ShortcutDef("open_project",       "File",     "Open Project",               "Ctrl+O"),
    ShortcutDef("save_project",       "File",     "Save Project",               "Ctrl+S"),
    ShortcutDef("save_as",            "File",     "Save Project As",            "Ctrl+Shift+S"),
    # Edit
    ShortcutDef("split",              "Edit",     "Split at Playhead",          "S"),
    ShortcutDef("delete",             "Edit",     "Delete Selected",            "W"),
    ShortcutDef("ripple_delete",      "Edit",     "Ripple Delete Selected",     "D"),
    ShortcutDef("undo",               "Edit",     "Undo",                       "Ctrl+Z"),
    ShortcutDef("redo",               "Edit",     "Redo",                       "Ctrl+Shift+Z"),
    ShortcutDef("select_all",         "Edit",     "Select All",                 "Ctrl+A"),
    ShortcutDef("select_to_gap",      "Edit",     "Select to Gap Left",         "A"),
    ShortcutDef("set_in",             "Edit",     "Set In Point",               "E"),
    ShortcutDef("set_out",            "Edit",     "Set Out Point",              "R"),
    ShortcutDef("clear_in_out",       "Edit",     "Clear In/Out",               "X"),
    ShortcutDef("selection_mode",     "Edit",     "Selection Mode",             "V"),
    ShortcutDef("cut_mode",           "Edit",     "Cut Mode",                   "C"),
    ShortcutDef("scrub_follow",       "Edit",     "Scrub Follow",               "F"),
    # Timeline
    ShortcutDef("import_video",       "Timeline", "Import Video",               "Ctrl+I"),
    ShortcutDef("export_video",       "Timeline", "Export Video",               "Ctrl+E"),
    ShortcutDef("export_images",      "Timeline", "Export Image Sequence",      "Ctrl+Shift+E"),
    ShortcutDef("detect_cuts",        "Timeline", "Detect Cuts",                "Ctrl+D"),
    ShortcutDef("play_pause",         "Timeline", "Play / Pause",               "Space"),
    ShortcutDef("playhead_left",      "Timeline", "Step Playhead Left",         "Left"),
    ShortcutDef("playhead_left_fast", "Timeline", "Step Playhead Left (10 frames)",   "Shift+Left"),
    ShortcutDef("playhead_right",     "Timeline", "Step Playhead Right",        "Right"),
    ShortcutDef("playhead_right_fast","Timeline", "Step Playhead Right (10 frames)",  "Shift+Right"),
    ShortcutDef("playhead_home",      "Timeline", "Go to Start",                "Home"),
    ShortcutDef("playhead_end",       "Timeline", "Go to End",                  "End"),
    ShortcutDef("select_next_clip",   "Timeline", "Select Next Clip",           "Down"),
    ShortcutDef("select_prev_clip",   "Timeline", "Select Previous Clip",       "Up"),
    # People — toggle group membership for the currently-selected clips.
    # If no group holds this digit yet, the handler prompts to create one
    # inline. Order matches the keyboard's number row: 1-9 first, 0 at end.
    ShortcutDef("group_digit_1",      "People",   "Toggle Group 1",             "1"),
    ShortcutDef("group_digit_2",      "People",   "Toggle Group 2",             "2"),
    ShortcutDef("group_digit_3",      "People",   "Toggle Group 3",             "3"),
    ShortcutDef("group_digit_4",      "People",   "Toggle Group 4",             "4"),
    ShortcutDef("group_digit_5",      "People",   "Toggle Group 5",             "5"),
    ShortcutDef("group_digit_6",      "People",   "Toggle Group 6",             "6"),
    ShortcutDef("group_digit_7",      "People",   "Toggle Group 7",             "7"),
    ShortcutDef("group_digit_8",      "People",   "Toggle Group 8",             "8"),
    ShortcutDef("group_digit_9",      "People",   "Toggle Group 9",             "9"),
    ShortcutDef("group_digit_0",      "People",   "Toggle Group 0",             "0"),
]

_QSETTINGS_PREFIX = "shortcuts/"


def _normalise(key: str) -> str:
    """Round-trip through QKeySequence so equivalent strings (e.g. 'CTRL+s'
    and 'Ctrl+S') compare equal in the conflict check."""
    if not key:
        return ""
    return QKeySequence(key).toString()


class ShortcutManager:
    """Live registry of customizable shortcuts.

    Lifecycle:
      1. Construct (loads from QSettings, falls back to defaults).
      2. As the UI builds menu actions and timeline shortcuts, call
         ``attach_action(sid, qaction)`` / ``attach_qshortcut(sid, qshortcut)``.
         Each attach applies the current key immediately.
      3. ``set_key`` / ``reset_one`` / ``reset_all`` mutate the bindings and
         push the new key out to every attached target. Persists to
         QSettings on every change.
    """

    def __init__(self):
        self._defs: Dict[str, ShortcutDef] = {d.id: d for d in SHORTCUTS}
        self._current: Dict[str, str] = {}
        self._actions: Dict[str, QAction] = {}
        self._qshortcuts: Dict[str, QShortcut] = {}
        self._load_from_settings()

    # ------------------------------------------------------------------
    # Public API

    def get(self, sid: str) -> str:
        return self._current.get(sid, "")

    def get_default(self, sid: str) -> str:
        return self._defs[sid].default

    def get_all(self) -> List[Tuple[ShortcutDef, str]]:
        """Returns every (definition, current_key) pair in registry order."""
        return [(d, self._current[d.id]) for d in SHORTCUTS]

    def attach_action(self, sid: str, action: QAction):
        """Register a QAction so future reassignments update its shortcut.
        The current key is applied to the action immediately."""
        self._actions[sid] = action
        action.setShortcut(QKeySequence(self._current[sid]))

    def attach_qshortcut(self, sid: str, qshortcut: QShortcut):
        """Register a QShortcut. Current key applied immediately."""
        self._qshortcuts[sid] = qshortcut
        qshortcut.setKey(QKeySequence(self._current[sid]))

    def set_key(self, sid: str, key: str) -> Optional[str]:
        """Try to assign ``key`` to ``sid``. Returns:
          - ``None`` on success (key applied + persisted).
          - The conflicting action's display name when ``key`` is already
            bound elsewhere (assignment NOT applied).
        Empty ``key`` means 'clear' and is always allowed.
        """
        norm = _normalise(key)
        if norm:
            for other_id, other_key in self._current.items():
                if other_id == sid:
                    continue
                if other_key == norm:
                    return self._defs[other_id].name
        self._current[sid] = norm
        self._persist(sid)
        self._apply(sid)
        return None

    def reset_one(self, sid: str) -> Optional[str]:
        """Restore one shortcut to its default. Same return contract as
        ``set_key`` — may fail if the default is in use elsewhere."""
        return self.set_key(sid, self._defs[sid].default)

    def reset_all(self):
        """Restore every shortcut to its default. Done in two passes so
        no transient conflict can fire mid-reset."""
        for d in SHORTCUTS:
            self._current[d.id] = ""
            self._persist(d.id)
            self._apply(d.id)
        for d in SHORTCUTS:
            self._current[d.id] = d.default
            self._persist(d.id)
            self._apply(d.id)

    # ------------------------------------------------------------------
    # Internal

    def _load_from_settings(self):
        """Populate ``self._current`` from QSettings; on collision (e.g. a
        user override clashes with a newer app version's default), the
        later entry is cleared so the invariant holds at startup."""
        s = QSettings()
        used: Dict[str, str] = {}  # key -> sid
        for d in SHORTCUTS:
            saved = s.value(_QSETTINGS_PREFIX + d.id, type=str)
            candidate = _normalise(saved) if saved else _normalise(d.default)
            if candidate and candidate in used:
                # Collision — keep the first owner, clear this one.
                self._current[d.id] = ""
            else:
                self._current[d.id] = candidate
                if candidate:
                    used[candidate] = d.id

    def _persist(self, sid: str):
        QSettings().setValue(_QSETTINGS_PREFIX + sid, self._current[sid])

    def _apply(self, sid: str):
        seq = QKeySequence(self._current[sid])
        if sid in self._actions:
            self._actions[sid].setShortcut(seq)
        if sid in self._qshortcuts:
            self._qshortcuts[sid].setKey(seq)
