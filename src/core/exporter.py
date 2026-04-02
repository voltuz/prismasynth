import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.timeline import TimelineModel
from core.video_source import VideoSource

logger = logging.getLogger(__name__)

# Cached GPU probe result (None = not yet probed)
_gpu_tonemap_available: Optional[bool] = None
_opencl_device: Optional[str] = None
_gpu_probe_lock = threading.Lock()


def _probe_gpu_tonemap() -> Tuple[bool, Optional[str]]:
    """Check if OpenCL tonemap is available. Returns (available, device_string)."""
    global _gpu_tonemap_available, _opencl_device
    with _gpu_probe_lock:
        if _gpu_tonemap_available is not None:
            return _gpu_tonemap_available, _opencl_device

        # Try to find a usable OpenCL device
        for device in ["0.0", "0.1", "1.0"]:
            try:
                cmd = [
                    "ffmpeg", "-v", "quiet",
                    "-init_hw_device", f"opencl=ocl:{device}",
                    "-filter_hw_device", "ocl",
                    "-f", "lavfi", "-i",
                    "smptehdbars=s=64x64:d=0.04,format=yuv420p10le,"
                    "setparams=color_trc=smpte2084:colorspace=bt2020nc:color_primaries=bt2020",
                    "-vf", "format=p010le,hwupload,"
                           "tonemap_opencl=tonemap=hable:format=nv12,"
                           "hwdownload,format=nv12",
                    "-f", "null", "-",
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode == 0:
                    _gpu_tonemap_available = True
                    _opencl_device = device
                    logger.info("GPU tonemap_opencl available on device %s", device)
                    return True, device
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        _gpu_tonemap_available = False
        _opencl_device = None
        logger.info("GPU tonemap_opencl not available, using CPU fallback")
        return False, None


class Exporter(QObject):
    """Exports the timeline as video (via FFmpeg) or image sequence."""

    progress = Signal(int)     # percentage 0-100
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, timeline: TimelineModel, sources: Dict[str, VideoSource],
                 parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        procs = getattr(self, '_active_procs', [])
        self._active_procs = []
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass

    def export(self, settings: dict):
        """Start export in a background thread."""
        thread = threading.Thread(target=self._run_export, args=(settings,), daemon=True)
        thread.start()

    def _build_segments(self) -> List[Tuple]:
        """Build list of (source_path, source_in, frame_count, source_fps, source_id)
        segments clipped to the render range."""
        clips = self._timeline.clips
        render_start, render_end = self._timeline.get_render_range()
        segments = []
        pos = 0
        for clip in clips:
            clip_start = pos
            clip_end = pos + clip.duration_frames - 1
            pos += clip.duration_frames
            if clip_end < render_start or clip_start > render_end:
                continue
            if clip.is_gap:
                continue
            source = self._sources.get(clip.source_id)
            if source is None:
                continue
            effective_start = max(clip_start, render_start)
            effective_end = min(clip_end, render_end)
            offset_in_clip = effective_start - clip_start
            src_in = clip.source_in + offset_in_clip
            frame_count = effective_end - effective_start + 1
            segments.append((source.file_path, src_in, frame_count, source.fps, clip.source_id))
        return segments

    def _build_vf(self, width: int, height: int, gpu_tonemap: bool,
                  output_format: str = None) -> str:
        """Build the ffmpeg video filter chain."""
        if gpu_tonemap:
            vf = (
                f"format=p010le,"
                f"hwupload,"
                f"tonemap_opencl=tonemap=hable:desat=0:peak=100:format=nv12,"
                f"hwdownload,format=nv12,"
                f"scale={width}:{height}:flags=lanczos"
            )
        else:
            vf = (
                f"zscale=t=linear:npl=100,format=gbrpf32le,"
                f"zscale=p=bt709,tonemap=hable:desat=0,"
                f"zscale=t=bt709:m=bt709:r=tv,"
                f"format=yuv420p,scale={width}:{height}:flags=lanczos"
            )
        if output_format:
            vf += f",format={output_format}"
        return vf

    def _gpu_hw_args(self, device: str) -> list:
        """FFmpeg args to initialize OpenCL device for GPU tonemap."""
        return ["-init_hw_device", f"opencl=ocl:{device}", "-filter_hw_device", "ocl"]

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

    def _export_video(self, settings: dict):
        width = settings["width"]
        height = settings["height"]
        fps = settings["fps"]
        output_path = settings["output_path"]
        ffmpeg_args = settings["ffmpeg_args"]

        segments = self._build_segments()
        if not segments:
            self.status.emit("Nothing to export")
            return

        gpu_available, opencl_device = _probe_gpu_tonemap()
        vf = self._build_vf(width, height, gpu_available)
        hw_args = self._gpu_hw_args(opencl_device) if gpu_available else []

        # Parallel segment encoding: GPU tonemap is per-process, so more processes = more utilization
        is_nvenc = any("nvenc" in a for a in ffmpeg_args)
        max_parallel = 6 if is_nvenc else 3

        total_frames = sum(s[2] for s in segments)
        self.status.emit(f"Encoding to {output_path}..." +
                         (" (GPU tonemap)" if gpu_available else ""))

        temp_dir = tempfile.mkdtemp(prefix="prismasynth_export_",
                                     dir=os.path.dirname(output_path))
        try:
            self._export_video_concat(
                segments, temp_dir, output_path, fps,
                vf, hw_args, ffmpeg_args, max_parallel, total_frames,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _export_video_concat(self, segments, temp_dir, output_path, fps,
                             vf, hw_args, ffmpeg_args, max_parallel, total_frames):
        """Encode each segment to a temp file, then concat."""
        temp_files = []
        commands = []

        for i, (source_path, src_in, duration, src_fps, _sid) in enumerate(segments):
            ss = src_in / src_fps if src_fps > 0 else 0
            dur = duration / src_fps if src_fps > 0 else 0
            tmp = os.path.join(temp_dir, f"seg_{i:04d}.mkv")
            temp_files.append(tmp)

            cmd = ["ffmpeg", "-y", "-v", "quiet"] + hw_args + [
                "-hwaccel", "cuda",
                "-ss", f"{ss:.6f}", "-i", source_path,
                "-t", f"{dur:.6f}",
                "-vf", vf,
                "-r", str(fps),
            ] + ffmpeg_args + [tmp]
            commands.append((cmd, duration))

        # Execute in batches
        frames_done = 0
        for batch_start in range(0, len(commands), max_parallel):
            if self._cancelled:
                return
            batch = commands[batch_start:batch_start + max_parallel]
            procs = []
            for cmd, _ in batch:
                procs.append(subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                ))
            self._active_procs = list(procs)

            # Wait for batch, checking for cancellation
            for p in procs:
                while p.poll() is None:
                    if self._cancelled:
                        for pp in procs:
                            try:
                                pp.kill()
                            except OSError:
                                pass
                        self._active_procs = []
                        return
                    try:
                        p.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            self._active_procs = []

            # Check for segment failures
            for idx_in_batch, p in enumerate(procs):
                if p.returncode != 0:
                    seg_idx = batch_start + idx_in_batch
                    stderr = p.stderr.read().decode(errors="replace")[-500:]
                    logger.error("Segment %d failed (code %d): %s", seg_idx, p.returncode, stderr)
                    self.error.emit(f"Segment {seg_idx} encoding failed")
                    return

            batch_frames = sum(dur for _, dur in batch)
            frames_done += batch_frames
            self.progress.emit(min(90, int(frames_done / total_frames * 90)))

        if self._cancelled:
            return

        # Concat all temp files into final output
        self.status.emit("Concatenating segments...")
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for tmp in temp_files:
                escaped = tmp.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy", output_path,
        ]
        result = subprocess.run(concat_cmd, capture_output=True)
        if result.returncode != 0 and not self._cancelled:
            stderr = result.stderr.decode(errors="replace")[-500:]
            logger.error("Concat failed: %s", stderr)
            self.status.emit(f"Concat error: {stderr}")
            return

        self.progress.emit(100)
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
        """Iterate frames using ffmpeg decode with HDR->SDR tone mapping."""
        segments = self._build_segments()
        total = sum(s[2] for s in segments)
        frame_count = 0
        frame_size = width * height * 3

        gpu_available, opencl_device = _probe_gpu_tonemap()
        vf = self._build_vf(width, height, gpu_available, output_format="rgb24")
        hw_args = self._gpu_hw_args(opencl_device) if gpu_available else []

        for source_path, source_in, duration, src_fps, _sid in segments:
            if self._cancelled:
                return

            ss = source_in / src_fps if src_fps > 0 else 0
            dur = duration / src_fps if src_fps > 0 else 0

            decode_cmd = [
                "ffmpeg", "-v", "quiet",
            ] + hw_args + [
                "-hwaccel", "cuda",
                "-ss", f"{ss:.6f}",
                "-i", source_path,
                "-t", f"{dur:.6f}",
                "-vf", vf,
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ]
            proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = bytearray()
            frames_read = 0
            while frames_read < duration and not self._cancelled:
                raw = proc.stdout.read(frame_size * 10)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= frame_size and frames_read < duration:
                    frame = np.frombuffer(bytes(buf[:frame_size]), np.uint8).reshape(height, width, 3)
                    del buf[:frame_size]
                    yield frame
                    frames_read += 1
                    frame_count += 1
                    if total > 0 and frame_count % max(1, total // 200) == 0:
                        self.progress.emit(min(99, int(frame_count / total * 100)))

            proc.stdout.close()
            proc.wait()

    @staticmethod
    def _save_exr(frame_rgb: np.ndarray, path: str):
        try:
            import imageio
            frame_float = frame_rgb.astype(np.float32) / 255.0
            imageio.imwrite(path, frame_float)
        except ImportError:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.replace(".exr", ".png"), bgr)
