import logging
import os
import subprocess
import threading
from typing import Dict

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.timeline import TimelineModel
from core.video_source import VideoSource
from core.video_reader import VideoReaderPool

logger = logging.getLogger(__name__)


class Exporter(QObject):
    """Exports the timeline as video (via FFmpeg pipe) or image sequence."""

    progress = Signal(int)     # percentage 0-100
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, timeline: TimelineModel, sources: Dict[str, VideoSource],
                 reader_pool: VideoReaderPool, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._reader_pool = reader_pool
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def export(self, settings: dict):
        """Start export in a background thread."""
        thread = threading.Thread(target=self._run_export, args=(settings,), daemon=True)
        thread.start()

    def _run_export(self, settings: dict):
        try:
            if settings["mode"] == "video":
                self._export_video(settings)
            elif settings["mode"] == "image_sequence":
                self._export_image_sequence(settings)
            if not self._cancelled:
                self.finished.emit()
        except Exception as e:
            logger.exception("Export failed")
            self.error.emit(str(e))
            self.status.emit(f"Error: {e}")

    def _iter_frames(self, width: int, height: int):
        """Iterate through all frames in timeline order, resized to (width, height)."""
        clips = self._timeline.clips
        # Total only counts real clip frames (not gaps) for accurate progress
        total = sum(c.duration_frames for c in clips if not c.is_gap)
        frame_count = 0

        for clip in clips:
            if self._cancelled:
                return
            if clip.is_gap:
                continue
            # Use playback reader to avoid blocking scrub reader
            reader = self._reader_pool.get_playback_reader(clip.source_id)
            if reader is None:
                continue
            for frame_data in reader.read_sequential(clip.source_in, clip.duration_frames):
                if self._cancelled:
                    return
                h, w = frame_data.shape[:2]
                if w != width or h != height:
                    frame_data = cv2.resize(frame_data, (width, height),
                                            interpolation=cv2.INTER_LANCZOS4)
                yield frame_data
                frame_count += 1
                if total > 0 and frame_count % max(1, total // 200) == 0:
                    pct = min(99, int(frame_count / total * 100))
                    self.progress.emit(pct)

    def _export_video(self, settings: dict):
        width = settings["width"]
        height = settings["height"]
        fps = settings["fps"]
        output_path = settings["output_path"]
        ffmpeg_args = settings["ffmpeg_args"]

        self.status.emit(f"Encoding to {output_path}...")

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ] + ffmpeg_args + [
            output_path
        ]

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        try:
            for frame_data in self._iter_frames(width, height):
                if self._cancelled:
                    proc.kill()
                    return
                proc.stdin.write(frame_data.tobytes())
        except BrokenPipeError:
            pass
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        if proc.returncode != 0 and not self._cancelled:
            stderr = proc.stderr.read().decode(errors="replace")
            logger.error("FFmpeg error: %s", stderr[-500:])
            self.status.emit(f"FFmpeg error (code {proc.returncode})")
        else:
            self.status.emit(f"Video saved to {output_path}")

    def _export_image_sequence(self, settings: dict):
        width = settings["width"]
        height = settings["height"]
        output_dir = settings["output_dir"]
        fmt = settings["format"]
        ext = settings["ext"]

        os.makedirs(output_dir, exist_ok=True)
        self.status.emit(f"Exporting frames to {output_dir}...")

        frame_index = 0
        for frame_data in self._iter_frames(width, height):
            if self._cancelled:
                return
            frame_index += 1
            filename = os.path.join(output_dir, f"{frame_index:06d}{ext}")

            if fmt == "exr":
                self._save_exr(frame_data, filename)
            elif fmt == "jpg":
                bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:  # png
                bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr)

        self.status.emit(f"Exported {frame_index} frames to {output_dir}")

    @staticmethod
    def _save_exr(frame_rgb: np.ndarray, path: str):
        try:
            import imageio
            # Convert uint8 RGB to float32 (0.0 - 1.0)
            frame_float = frame_rgb.astype(np.float32) / 255.0
            imageio.imwrite(path, frame_float)
        except ImportError:
            # Fallback: save as PNG if imageio not available
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.replace(".exr", ".png"), bgr)
