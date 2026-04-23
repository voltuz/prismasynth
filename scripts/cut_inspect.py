"""Cut Inspect — GUI debug tool.

Load a video (your PrismaSynth export), visually scrub to each cut
boundary and mark it, then extract the 5 frames around every cut
(N-2, N-1, N, N+1, N+2) plus frame 0 and the last frame. Inspect the
resulting PNGs to identify duplicates, misalignments, tonemap drops, or
whatever else is going wrong at segment boundaries.

Run:
    venv\\Scripts\\python scripts\\cut_inspect.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# libmpv-2.dll lives in src/
os.environ["PATH"] = str(SRC) + os.pathsep + os.environ.get("PATH", "")

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

import mpv


class VideoPreview(QWidget):
    """Minimal mpv-embedded preview for frame-accurate inspection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #000;")
        self._player: mpv.MPV | None = None
        self._fps: float = 0.0
        self._total_frames: int = 0

    def showEvent(self, event):
        super().showEvent(event)
        if self._player is None:
            self._init_player()

    def _init_player(self):
        wid = str(int(self.winId()))
        self._player = mpv.MPV(
            wid=wid,
            hwdec='auto',
            hr_seek='yes',
            hr_seek_framedrop='yes',
            keep_open='yes',
            keep_open_pause='yes',
            osd_level=0,
            cursor_autohide='no',
            input_cursor='no',
            input_default_bindings='no',
            input_vo_keyboard='no',
            ao='null',
        )
        self._player.pause = True

    def load(self, path: str) -> bool:
        if self._player is None:
            self._init_player()
        self._player.loadfile(path, 'replace')
        try:
            self._player.wait_for_property('seekable', timeout=10.0)
        except Exception:
            pass
        try:
            fps = self._player.container_fps
            dur = self._player.duration
            if fps and dur:
                self._fps = float(fps)
                self._total_frames = int(round(float(dur) * self._fps))
        except Exception:
            pass
        self._player.pause = True
        return self._total_frames > 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def seek_frame(self, frame: int):
        if self._player is None or self._fps <= 0:
            return
        self._player.pause = True
        ts = frame / self._fps
        self._player.command_async('seek', str(ts), 'absolute+exact')

    def step_forward(self):
        if self._player is None:
            return
        self._player.pause = True
        self._player.command('frame-step')

    def step_back(self):
        if self._player is None:
            return
        self._player.pause = True
        self._player.command('frame-back-step')

    def play_pause(self):
        if self._player is None:
            return
        self._player.pause = not self._player.pause

    def current_frame(self) -> int:
        if self._player is None or self._fps <= 0:
            return 0
        try:
            t = self._player.time_pos
            if t is None:
                return 0
            return int(round(float(t) * self._fps))
        except Exception:
            return 0

    def cleanup(self):
        if self._player is not None:
            try:
                self._player.terminate()
            except Exception:
                pass
            self._player = None


class CutInspect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cut Inspect")
        self.resize(1000, 780)

        self._video_path: str | None = None
        self._cuts: list[int] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File picker
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Video:"))
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        file_row.addWidget(self._file_edit, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_video)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        # Video preview
        self._preview = VideoPreview()
        layout.addWidget(self._preview, 1)

        # Transport / scrubber
        nav_row = QHBoxLayout()
        btn_back = QPushButton("◀ Step")
        btn_back.clicked.connect(self._preview.step_back)
        btn_play = QPushButton("Play / Pause")
        btn_play.clicked.connect(self._preview.play_pause)
        btn_fwd = QPushButton("Step ▶")
        btn_fwd.clicked.connect(self._preview.step_forward)
        nav_row.addWidget(btn_back)
        nav_row.addWidget(btn_play)
        nav_row.addWidget(btn_fwd)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setTracking(False)
        self._slider.sliderReleased.connect(self._on_slider_released)
        nav_row.addWidget(self._slider, 1)
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setMinimumWidth(140)
        nav_row.addWidget(self._frame_label)
        layout.addLayout(nav_row)

        # Mark controls
        mark_row = QHBoxLayout()
        btn_mark = QPushButton("Mark Cut at Current Frame")
        btn_mark.clicked.connect(self._mark_cut)
        mark_row.addWidget(btn_mark)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected)
        mark_row.addWidget(btn_remove)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_cuts)
        mark_row.addWidget(btn_clear)
        layout.addLayout(mark_row)

        # Cut list
        layout.addWidget(QLabel("Cut positions (first frame of the new clip):"))
        self._cut_list = QListWidget()
        self._cut_list.setMaximumHeight(120)
        self._cut_list.itemDoubleClicked.connect(self._jump_to_cut)
        layout.addWidget(self._cut_list)

        # Output picker
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_edit = QLineEdit()
        out_row.addWidget(self._out_edit, 1)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        # Extract
        self._extract_btn = QPushButton("Extract Debug Frames")
        self._extract_btn.clicked.connect(self._extract)
        layout.addWidget(self._extract_btn)

        # Status
        self._status = QLabel("No video loaded.")
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        # Keep the slider / frame readout in sync with mpv
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(100)

    # --- event handlers ---

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select exported video", "",
            "Video (*.mov *.mp4 *.mkv *.avi);;All files (*)")
        if not path:
            return
        self._video_path = path
        self._file_edit.setText(path)
        if not self._preview.load(path):
            QMessageBox.critical(self, "Load failed",
                                 "Could not determine fps / frame count.")
            return
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, self._preview.total_frames - 1))
        self._status.setText(
            f"Loaded: {self._preview.total_frames} frames @ "
            f"{self._preview.fps:.3f} fps")
        # Default output dir next to video
        if not self._out_edit.text():
            self._out_edit.setText(
                str(Path(path).with_suffix("").as_posix() + "_debug_frames"))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_edit.setText(path)

    def _on_slider_released(self):
        self._preview.seek_frame(self._slider.value())

    def _tick(self):
        if self._video_path is None:
            return
        f = self._preview.current_frame()
        total = self._preview.total_frames
        self._frame_label.setText(f"Frame: {f} / {total - 1 if total else 0}")
        if not self._slider.isSliderDown():
            self._slider.blockSignals(True)
            self._slider.setValue(f)
            self._slider.blockSignals(False)

    def _mark_cut(self):
        if self._video_path is None:
            return
        f = self._preview.current_frame()
        if f in self._cuts:
            return
        self._cuts.append(f)
        self._cuts.sort()
        self._refresh_cut_list()

    def _remove_selected(self):
        for item in self._cut_list.selectedItems():
            frame = int(item.data(Qt.ItemDataRole.UserRole))
            if frame in self._cuts:
                self._cuts.remove(frame)
        self._refresh_cut_list()

    def _clear_cuts(self):
        self._cuts.clear()
        self._refresh_cut_list()

    def _refresh_cut_list(self):
        self._cut_list.clear()
        for c in self._cuts:
            item = QListWidgetItem(f"Frame {c}")
            item.setData(Qt.ItemDataRole.UserRole, c)
            self._cut_list.addItem(item)

    def _jump_to_cut(self, item: QListWidgetItem):
        frame = int(item.data(Qt.ItemDataRole.UserRole))
        self._slider.setValue(frame)
        self._preview.seek_frame(frame)

    # --- extraction ---

    def _extract(self):
        if not self._video_path:
            QMessageBox.warning(self, "No video", "Load a video first.")
            return
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output", "Choose an output folder.")
            return

        # Build unique frame list: 0, cut±2 for each cut, last
        total = self._preview.total_frames
        frames: set[int] = {0}
        for c in self._cuts:
            for off in (-2, -1, 0, 1, 2):
                n = c + off
                if 0 <= n < total:
                    frames.add(n)
        if total > 0:
            frames.add(total - 1)
        if not frames:
            QMessageBox.warning(self, "Nothing to extract",
                                "No cut marks set.")
            return
        frames_sorted = sorted(frames)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Extracting frames…", "Cancel",
                                   0, len(frames_sorted), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        failed: list[int] = []
        for i, n in enumerate(frames_sorted):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Extracting frame {n}…")
            progress.setValue(i)
            QApplication.processEvents()
            ok = self._extract_one(n, out_path)
            if not ok:
                failed.append(n)
        progress.setValue(len(frames_sorted))

        msg = f"Extracted {len(frames_sorted) - len(failed)} / {len(frames_sorted)} frames."
        if failed:
            msg += f"\nFailed: {failed[:20]}"
        msg += f"\n\nOutput: {out_path}"
        QMessageBox.information(self, "Done", msg)

    def _extract_one(self, frame: int, out_dir: Path) -> bool:
        """Frame-exact extraction via a select filter. Decodes from start,
        but decode is fast on the exported (SDR, reasonable GOP) output."""
        out_file = out_dir / f"frame_{frame:06d}.png"
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-v", "error",
            "-i", self._video_path,
            "-vf", f"select=eq(n\\,{frame})",
            "-frames:v", "1",
            "-vsync", "vfr",
            str(out_file),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            return r.returncode == 0 and out_file.exists()
        except Exception:
            return False

    def closeEvent(self, event):
        self._preview.cleanup()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CutInspect()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
