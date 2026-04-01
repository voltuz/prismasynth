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

        # Build concat list of clip segments for ffmpeg
        clips = self._timeline.clips
        segments = []
        for clip in clips:
            if clip.is_gap:
                continue
            source = self._sources.get(clip.source_id)
            if source is None:
                continue
            segments.append((source.file_path, clip.source_in, clip.duration_frames, source.fps))

        if not segments:
            self.status.emit("Nothing to export")
            return

        # Export each segment via ffmpeg with proper HDR→SDR tone mapping
        # Use a concat approach: encode each segment to a temp file, then concat
        # Or simpler: pipe raw frames with tone mapping via ffmpeg per-segment
        total_frames = sum(s[2] for s in segments)
        frame_count = 0

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
            stderr=subprocess.DEVNULL
        )

        try:
            for source_path, source_in, duration, src_fps in segments:
                if self._cancelled:
                    break
                # Decode each segment with ffmpeg (handles HDR/DV tone mapping correctly)
                ss = source_in / src_fps if src_fps > 0 else 0
                dur = duration / src_fps if src_fps > 0 else 0
                decode_cmd = [
                    "ffmpeg", "-v", "quiet",
                    "-hwaccel", "cuda",
                    "-ss", f"{ss:.4f}",
                    "-i", source_path,
                    "-t", f"{dur:.4f}",
                    "-vf", (
                        f"zscale=t=linear:npl=100,format=gbrpf32le,"
                        f"zscale=p=bt709,tonemap=hable:desat=0,"
                        f"zscale=t=bt709:m=bt709:r=tv,"
                        f"format=yuv420p,scale={width}:{height}:flags=lanczos,"
                        f"format=rgb24"
                    ),
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "pipe:1",
                ]
                decode_proc = subprocess.Popen(
                    decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                frame_size = width * height * 3
                frames_read = 0
                buf = bytearray()
                while frames_read < duration and not self._cancelled:
                    raw = decode_proc.stdout.read(frame_size * 10)
                    if not raw:
                        break
                    buf.extend(raw)
                    while len(buf) >= frame_size and frames_read < duration:
                        proc.stdin.write(bytes(buf[:frame_size]))
                        del buf[:frame_size]
                        frames_read += 1
                        frame_count += 1
                        if frame_count % max(1, total_frames // 200) == 0:
                            self.progress.emit(min(99, int(frame_count / total_frames * 100)))

                decode_proc.stdout.close()
                decode_proc.kill()
                decode_proc.wait()

        except BrokenPipeError:
            pass
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        if proc.returncode != 0 and not self._cancelled:
            logger.error("FFmpeg exited with code %d", proc.returncode)
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
        for frame_data in self._iter_frames_ffmpeg(width, height):
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

    def _iter_frames_ffmpeg(self, width: int, height: int):
        """Iterate frames using ffmpeg decode with HDR→SDR tone mapping."""
        clips = self._timeline.clips
        total = sum(c.duration_frames for c in clips if not c.is_gap)
        frame_count = 0
        frame_size = width * height * 3

        for clip in clips:
            if self._cancelled:
                return
            if clip.is_gap:
                continue
            source = self._sources.get(clip.source_id)
            if source is None:
                continue

            ss = clip.source_in / source.fps if source.fps > 0 else 0
            dur = clip.duration_frames / source.fps if source.fps > 0 else 0

            decode_cmd = [
                "ffmpeg", "-v", "quiet",
                "-hwaccel", "cuda",
                "-ss", f"{ss:.4f}",
                "-i", source.file_path,
                "-t", f"{dur:.4f}",
                "-vf", (
                    f"zscale=t=linear:npl=100,format=gbrpf32le,"
                    f"zscale=p=bt709,tonemap=hable:desat=0,"
                    f"zscale=t=bt709:m=bt709:r=tv,"
                    f"format=yuv420p,scale={width}:{height}:flags=lanczos,"
                    f"format=rgb24"
                ),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ]
            proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = bytearray()
            frames_read = 0
            while frames_read < clip.duration_frames and not self._cancelled:
                raw = proc.stdout.read(frame_size * 10)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= frame_size and frames_read < clip.duration_frames:
                    frame = np.frombuffer(bytes(buf[:frame_size]), np.uint8).reshape(height, width, 3)
                    del buf[:frame_size]
                    yield frame
                    frames_read += 1
                    frame_count += 1
                    if total > 0 and frame_count % max(1, total // 200) == 0:
                        self.progress.emit(min(99, int(frame_count / total * 100)))

            proc.stdout.close()
            proc.kill()
            proc.wait()

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
