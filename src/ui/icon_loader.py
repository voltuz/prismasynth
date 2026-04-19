import os

from PySide6.QtGui import QIcon

_ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
_cache: dict[str, QIcon] = {}


def icon(name: str) -> QIcon:
    """Load (and cache) an SVG icon from src/ui/icons/{name}.svg."""
    cached = _cache.get(name)
    if cached is not None:
        return cached
    path = os.path.join(_ICONS_DIR, f"{name}.svg")
    ic = QIcon(path) if os.path.exists(path) else QIcon()
    _cache[name] = ic
    return ic
