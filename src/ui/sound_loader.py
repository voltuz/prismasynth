"""Completion-sound playback for long-running operations.

Mirrors ``ui.icon_loader``: bundled assets live next to this module, here
under ``src/ui/sounds/``. PySide6 ships QtMultimedia, so no extra
dependency is needed.

API:
    init(parent)              -- call once on app startup with MainWindow.
    play_completion(window)   -- call from any "operation finished" slot.
    is_enabled() / set_enabled(b)
        -- the View menu toggle reads/writes through these.

Behaviour:
    - Toggle persists in QSettings (``notifications/completion_sound``,
      default True).
    - Plays only when the supplied ``window`` is not the active window
      (suppress while the user is right there watching the dialog).
    - All audio failures (missing file, no device, codec init error)
      degrade to a no-op + log warning; never raise.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PySide6.QtCore import QObject, QSettings, QUrl
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


_SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
_COMPLETE_WAV = os.path.join(_SOUNDS_DIR, "complete.wav")
_QSETTINGS_KEY = "notifications/completion_sound"

_effect: Optional[QSoundEffect] = None


def init(parent: QObject) -> None:
    """Construct the cached QSoundEffect. Must be called on the UI thread
    once the QApplication is running. ``parent`` keeps the effect alive."""
    global _effect
    if _effect is not None:
        return
    if not os.path.exists(_COMPLETE_WAV):
        logger.warning("Completion sound missing: %s (sound disabled)",
                       _COMPLETE_WAV)
        return
    try:
        _effect = QSoundEffect(parent)
        _effect.setSource(QUrl.fromLocalFile(_COMPLETE_WAV))
        _effect.setVolume(0.7)
    except Exception:
        logger.exception("Failed to initialise completion sound")
        _effect = None


def is_enabled() -> bool:
    raw = QSettings().value(_QSETTINGS_KEY, True)
    # QSettings on Windows can return the string 'true'/'false' rather
    # than a real bool depending on registry typing.
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "0", "no", "off")
    return bool(raw)


def set_enabled(enabled: bool) -> None:
    QSettings().setValue(_QSETTINGS_KEY, bool(enabled))


def play_completion(window: Optional[QWidget] = None) -> None:
    """Play the completion ding if the toggle is on and the user has
    switched away from the main window. Safe no-op otherwise."""
    if _effect is None:
        return
    if not is_enabled():
        return
    if window is not None and window.isActiveWindow():
        return
    try:
        _effect.play()
    except Exception:
        logger.exception("QSoundEffect.play() raised")
