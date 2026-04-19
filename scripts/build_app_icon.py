"""Render src/ui/icons/app.svg into src/app.ico with multiple sizes.
Run after editing the SVG:  venv\\Scripts\\python scripts\\build_app_icon.py
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_PATH = os.path.join(ROOT, "src", "ui", "icons", "app.svg")
ICO_PATH = os.path.join(ROOT, "src", "app.ico")
SIZES = [16, 24, 32, 48, 64, 128, 256]

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QImage, QPainter, QGuiApplication
from PySide6.QtSvg import QSvgRenderer
from PIL import Image


def render(svg_path: str, size: int) -> Image.Image:
    renderer = QSvgRenderer(svg_path)
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()
    buf = QByteArray()
    from PySide6.QtCore import QBuffer
    qbuf = QBuffer(buf)
    qbuf.open(QBuffer.OpenModeFlag.WriteOnly)
    img.save(qbuf, "PNG")
    return Image.open(io.BytesIO(bytes(buf)))


def main():
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    pngs = [render(SVG_PATH, s) for s in SIZES]
    pngs[-1].save(ICO_PATH, format="ICO", sizes=[(s, s) for s in SIZES], append_images=pngs[:-1])
    print(f"wrote {ICO_PATH} with sizes {SIZES}")


if __name__ == "__main__":
    main()
