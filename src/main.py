import sys
import os
import logging
import platform
import traceback

# Add src to path so imports work as packages
_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)
# Ensure libmpv-2.dll is findable
os.environ['PATH'] = _src_dir + os.pathsep + os.environ['PATH']

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont

from ui.main_window import MainWindow
from core.ui_scale import ui_scale

_CRASH_LOG = os.path.join(_src_dir, "crash.log")


def _excepthook(exc_type, exc_value, exc_tb):
    """Write unhandled exceptions to crash.log so they survive .bat launches."""
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"\n--- {datetime.now().isoformat()} ---\n{text}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    sys.excepthook = _excepthook
    setup_logging()

    # Windows: declare an explicit AppUserModelID so the taskbar uses our
    # own icon/grouping instead of inheriting Python's.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "prismasynth.app")
        except Exception:
            pass

    from version import __version__

    # Startup banner — shows what's running in the console (run.bat keeps the
    # cmd window open). Headline is name + version; the rest is environment
    # context for bug reports. Version lookups are defensive so a packaging
    # quirk can never block launch.
    _log = logging.getLogger("prismasynth")
    _log.info("=" * 56)
    _log.info("PrismaSynth v%s", __version__)
    try:
        from PySide6 import __version__ as _pyside_version
        from PySide6.QtCore import qVersion
        _log.info("Python %s  |  PySide6 %s (Qt %s)",
                  platform.python_version(), _pyside_version, qVersion())
    except Exception:
        _log.info("Python %s", platform.python_version())
    _log.info("Platform: %s", platform.platform())
    _log.info("=" * 56)

    app = QApplication(sys.argv)
    app.setApplicationName("PrismaSynth")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    # Capture the system base font BEFORE any scaling so live re-scales
    # always derive from the same anchor (re-scaling a derived font would
    # compound).
    _base_font = QFont(app.font())
    _base_pt = _base_font.pointSize() if _base_font.pointSize() > 0 else 9

    def _apply_app_font():
        f = QFont(_base_font)
        f.setPointSize(ui_scale().font_pt(_base_pt))
        app.setFont(f)

    _apply_app_font()
    ui_scale().changed.connect(_apply_app_font)

    icon_path = os.path.join(_src_dir, "app.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
