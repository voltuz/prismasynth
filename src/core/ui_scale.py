"""Global UI scale singleton.

Holds the user's chosen UI scale percent (75 / 100 / 125 / 150 / 200 by
default), persists it to QSettings under ``ui/scale_percent``, and emits
``changed`` whenever the user picks a new value. Persistent widgets
(timeline strip, side panels, preview, main window) connect to that
signal and re-apply their scaled sizes; modal dialogs simply read the
current scale at construction time.

Usage:

    from core.ui_scale import ui_scale

    # In a widget's __init__:
    s = ui_scale()
    self._swatch.setFixedSize(s.px(24), s.px(24))
    s.changed.connect(self._refresh_scale)

    # Anywhere — convert design pixels / point sizes to scaled values:
    ui_scale().px(14)        # int pixels at the current scale
    ui_scale().font_pt(8)    # int point size at the current scale
"""

from PySide6.QtCore import QObject, QSettings, Signal


class UIScale(QObject):
    changed = Signal()
    PRESETS = (75, 100, 125, 150, 200)
    _MIN, _MAX = 50, 300
    _SETTINGS_KEY = "ui/scale_percent"

    _instance = None

    @classmethod
    def instance(cls) -> "UIScale":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        v = QSettings().value(self._SETTINGS_KEY, 100)
        try:
            p = int(v) if v is not None else 100
        except (TypeError, ValueError):
            p = 100
        self._percent = max(self._MIN, min(self._MAX, p))

    @property
    def percent(self) -> int:
        return self._percent

    @property
    def factor(self) -> float:
        return self._percent / 100.0

    def set_percent(self, percent: int) -> None:
        p = max(self._MIN, min(self._MAX, int(percent)))
        if p == self._percent:
            return
        self._percent = p
        QSettings().setValue(self._SETTINGS_KEY, p)
        self.changed.emit()

    def px(self, value) -> int:
        return max(1, int(round(float(value) * self.factor)))

    def font_pt(self, pt) -> int:
        return max(6, int(round(float(pt) * self.factor)))


def ui_scale() -> UIScale:
    return UIScale.instance()
