import logging
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.timeline import TimelineModel
from core.video_source import VideoSource
from utils.ffprobe import probe_hdr

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

    def _build_source_groups(self):
        """Group segments by source and coalesce contiguous clips."""
        segments = self._build_segments()
        if not segments:
            return {}
        groups = {}
        for path, src_in, count, fps, sid in segments:
            if sid not in groups:
                groups[sid] = {"path": path, "fps": fps, "segments": [],
                               "total_frames": 0}
            groups[sid]["segments"].append((src_in, count))
            groups[sid]["total_frames"] += count
        # Coalesce contiguous segments within each source
        for g in groups.values():
            g["segments"].sort(key=lambda s: s[0])
            coalesced = [list(g["segments"][0])]
            for src_in, count in g["segments"][1:]:
                prev_in, prev_count = coalesced[-1]
                if src_in == prev_in + prev_count:
                    coalesced[-1][1] += count
                else:
                    coalesced.append([src_in, count])
            g["segments"] = [(s[0], s[1]) for s in coalesced]
        return groups

    @staticmethod
    def _build_select_expr(segments, seek_frame):
        """Build FFmpeg select filter expression for given segments."""
        terms = []
        for src_in, count in segments:
            a = src_in - seek_frame
            b = a + count - 1
            terms.append(f"between(n\\,{a}\\,{b})")
        return "select='" + "+".join(terms) + "'"

    def _build_vf(self, width: int, height: int, gpu_tonemap: bool,
                  output_format: str = None, is_hdr: bool = True,
                  source_width: int = 0, source_height: int = 0,
                  select_expr: str = None, fps: float = 0) -> Optional[str]:
        """Build the ffmpeg video filter chain. Returns None if no filters needed."""
        parts = []
        # Select filter for single-process-per-source mode
        if select_expr:
            parts.append(select_expr)
            if fps > 0:
                parts.append(f"setpts=N/{fps}/TB")
        need_scale = not (source_width > 0 and source_height > 0
                          and width == source_width and height == source_height)
        if is_hdr:
            if gpu_tonemap:
                parts.extend([
                    "format=p010le", "hwupload",
                    "tonemap_opencl=tonemap=hable:desat=0:peak=100:format=nv12",
                    "hwdownload", "format=nv12",
                ])
            else:
                parts.extend([
                    "zscale=t=linear:npl=100", "format=gbrpf32le",
                    "zscale=p=bt709", "tonemap=hable:desat=0",
                    "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p",
                ])
            if need_scale:
                parts.append(f"scale={width}:{height}:flags=lanczos")
        else:
            # SDR: no tonemap needed
            if need_scale:
                parts.append(f"scale={width}:{height}:flags=lanczos")
        if output_format:
            parts.append(f"format={output_format}")
        return ",".join(parts) if parts else None

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
        # All codecs benefit from parallel segment encoding:
        # - NVENC: exploits multiple HW decode/encode units
        # - CPU codecs (ProRes, x264, etc.): multiple processes saturate all cores
        try:
            self._export_video_parallel(settings)
        except Exception as e:
            if self._cancelled:
                return
            logger.exception("Parallel export failed, falling back to legacy")
            self.status.emit(f"Parallel failed: {e} — retrying (legacy)...")
            self.progress.emit(0)
            self._export_video_concat_legacy(settings)

    def _export_video_parallel(self, settings: dict):
        """Enhanced parallel export: coalesced segments, HDR-aware, thread pool."""
        width = settings["width"]
        height = settings["height"]
        fps = settings["fps"]
        output_path = settings["output_path"]
        ffmpeg_args = settings["ffmpeg_args"]

        groups = self._build_source_groups()
        if not groups:
            self.status.emit("Nothing to export")
            return

        gpu_available, opencl_device = _probe_gpu_tonemap()
        is_nvenc = any("nvenc" in a for a in ffmpeg_args)
        # NVENC: 6 parallel (GPU has dedicated HW units)
        # CPU codecs: scale with core count (each process is CPU-bound)
        cpu_count = os.cpu_count() or 8
        max_parallel = 6 if is_nvenc else max(2, cpu_count // 4)

        # Flatten coalesced segments back out with per-source VF chains
        flat_segments = []  # [(source_path, src_in, count, fps, vf, hw_args)]
        for sid, group in groups.items():
            source = self._sources.get(sid)
            is_hdr = probe_hdr(group["path"])
            use_gpu_tm = gpu_available and is_hdr
            hw = self._gpu_hw_args(opencl_device) if use_gpu_tm else []
            vf = self._build_vf(
                width, height, use_gpu_tm, is_hdr=is_hdr,
                source_width=source.width if source else 0,
                source_height=source.height if source else 0,
            )
            for src_in, count in group["segments"]:
                flat_segments.append(
                    (group["path"], src_in, count, group["fps"], vf, hw,
                     is_hdr))

        total_frames = sum(s[2] for s in flat_segments)
        suffix = ""
        if any(probe_hdr(g["path"]) for g in groups.values()):
            suffix = " (GPU tonemap)" if gpu_available else " (CPU tonemap)"
        self.status.emit(
            f"Encoding {len(flat_segments)} segments to {output_path}...{suffix}")

        def _build_segment_cmd(path, src_in, count, src_fps, vf, hw, hdr,
                               out_file):
            """Build FFmpeg command for a single segment."""
            ss = src_in / src_fps if src_fps > 0 else 0
            hwaccel = ["-hwaccel", "cuda"]
            # SDR + NVENC + no filters: full GPU pipeline (zero-copy)
            if not vf and is_nvenc:
                hwaccel += ["-hwaccel_output_format", "cuda"]
                gpu_vf = "scale_cuda=format=yuv420p"
            else:
                gpu_vf = None
            cmd = (["ffmpeg", "-y", "-nostdin", "-v", "error"] + hw
                   + hwaccel + [
                "-ss", f"{ss:.6f}",
                "-i", path,
                "-frames:v", str(count),
            ])
            if vf:
                cmd += ["-vf", vf]
            elif gpu_vf:
                cmd += ["-vf", gpu_vf]
            if hdr:
                cmd += ["-colorspace", "bt709", "-color_trc", "bt709",
                        "-color_primaries", "bt709"]
            if not is_nvenc:
                threads_per = max(2, cpu_count // max_parallel)
                cmd += ["-threads", str(threads_per)]
            cmd += ["-r", str(fps), "-an"] + ffmpeg_args + [out_file]
            return cmd

        # Single segment: encode directly to output (no temp files, no concat)
        if len(flat_segments) == 1:
            path, src_in, count, src_fps, vf, hw, hdr = flat_segments[0]
            cmd = _build_segment_cmd(path, src_in, count, src_fps, vf, hw,
                                     hdr, output_path)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._active_procs = [proc]
            _, stderr_data = proc.communicate()
            self._active_procs = []
            if proc.returncode != 0 and not self._cancelled:
                err = (stderr_data or b"").decode(errors="replace")[-500:]
                raise RuntimeError(f"Encoding failed: {err}")
            self.progress.emit(100)
            self.status.emit(f"Video saved to {output_path}")
            return

        # Multiple segments: temp files + parallel encode + concat
        temp_dir = tempfile.mkdtemp(prefix="prismasynth_export_",
                                     dir=os.path.dirname(output_path))
        try:
            temp_files = []
            commands = []
            for i, (path, src_in, count, src_fps, vf, hw, hdr) in enumerate(
                    flat_segments):
                tmp = os.path.join(temp_dir, f"seg_{i:04d}.mkv")
                temp_files.append(tmp)
                cmd = _build_segment_cmd(path, src_in, count, src_fps, vf, hw,
                                         hdr, tmp)
                commands.append((cmd, count))

            # Thread pool: as-soon-as-done scheduling (no batch-wait)
            frames_done = 0
            procs_lock = threading.Lock()
            active_procs = []
            failed = threading.Event()

            def _encode_segment(idx):
                if self._cancelled or failed.is_set():
                    return idx, -1, b"cancelled"
                cmd, _ = commands[idx]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                with procs_lock:
                    active_procs.append(proc)
                _, stderr_data = proc.communicate()
                with procs_lock:
                    if proc in active_procs:
                        active_procs.remove(proc)
                return idx, proc.returncode, stderr_data or b""

            def _kill_active():
                with procs_lock:
                    for p in active_procs:
                        try:
                            p.kill()
                        except OSError:
                            pass
                    active_procs.clear()

            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                futures = {pool.submit(_encode_segment, i): i
                           for i in range(len(commands))}
                try:
                    for future in as_completed(futures):
                        if self._cancelled:
                            _kill_active()
                            return
                        idx, rc, stderr = future.result()
                        if rc != 0 and not self._cancelled:
                            failed.set()
                            _kill_active()
                            err = stderr.decode(errors="replace")[-500:]
                            logger.error("Segment %d failed (code %d): %s",
                                         idx, rc, err)
                            raise RuntimeError(
                                f"Segment {idx} encoding failed: {err}")
                        frames_done += commands[idx][1]
                        self.progress.emit(
                            min(90, int(frames_done / total_frames * 90)))
                finally:
                    self._active_procs = []

            if self._cancelled:
                return

            # Concat
            self.status.emit("Concatenating segments...")
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for tmp in temp_files:
                    escaped = tmp.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            concat_cmd = [
                "ffmpeg", "-y", "-nostdin", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c", "copy", output_path,
            ]
            result = subprocess.run(concat_cmd, capture_output=True)
            if result.returncode != 0 and not self._cancelled:
                stderr = result.stderr.decode(errors="replace")[-500:]
                raise RuntimeError(f"Concat failed: {stderr}")

            self.progress.emit(100)
            self.status.emit(f"Video saved to {output_path}")
        finally:
            self._active_procs = []
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _export_video_concat_legacy(self, settings: dict):
        """Legacy fallback: one FFmpeg process per segment, then concat."""
        width = settings["width"]
        height = settings["height"]
        fps = settings["fps"]
        output_path = settings["output_path"]
        ffmpeg_args = settings["ffmpeg_args"]

        segments = self._build_segments()
        if not segments:
            self.status.emit("Nothing to export")
            return

        # Detect source dimensions for skip-scale optimization
        first_sid = segments[0][4]
        first_source = self._sources.get(first_sid)
        src_w = first_source.width if first_source else 0
        src_h = first_source.height if first_source else 0

        gpu_available, opencl_device = _probe_gpu_tonemap()
        vf = self._build_vf(width, height, gpu_available,
                            source_width=src_w, source_height=src_h)
        hw_args = self._gpu_hw_args(opencl_device) if gpu_available else []

        is_nvenc = any("nvenc" in a for a in ffmpeg_args)
        max_parallel = 6 if is_nvenc else 3

        total_frames = sum(s[2] for s in segments)
        self.status.emit(f"Encoding to {output_path} (legacy)..." +
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
            tmp = os.path.join(temp_dir, f"seg_{i:04d}.mkv")
            temp_files.append(tmp)

            cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"] + hw_args + [
                "-hwaccel", "cuda",
                "-ss", f"{ss:.6f}",
                "-i", source_path,
                "-frames:v", str(duration),
                "-vf", vf,
                "-r", str(fps), "-an",
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
            "ffmpeg", "-y", "-nostdin", "-v", "error",
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

        def _write_frame(index, data):
            filename = os.path.join(output_dir, f"{index:06d}{ext}")
            if fmt == "exr":
                self._save_exr(data, filename)
            elif fmt == "jpg":
                bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:  # png
                bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr)

        frame_index = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = []
            for frame_data in self._iter_frames_ffmpeg(width, height):
                if self._cancelled:
                    return
                frame_index += 1
                # Submit write to thread pool (overlaps I/O with decode)
                pending.append(pool.submit(_write_frame, frame_index,
                                           frame_data.copy()))
                # Limit queue depth to avoid memory bloat
                if len(pending) >= 16:
                    pending[0].result()
                    pending.pop(0)
            # Drain remaining writes
            for f in pending:
                f.result()

        self.status.emit(f"Exported {frame_index} frames to {output_dir}")

    def _iter_frames_ffmpeg(self, width: int, height: int):
        """Iterate frames using ffmpeg decode with HDR->SDR tone mapping.
        Uses single-process-per-source with select filter for efficiency."""
        groups = self._build_source_groups()
        if not groups:
            return
        total = sum(g["total_frames"] for g in groups.values())
        frame_count = 0
        frame_size = width * height * 3

        gpu_available, opencl_device = _probe_gpu_tonemap()

        for sid, group in groups.items():
            if self._cancelled:
                return

            source = self._sources.get(sid)
            is_hdr = probe_hdr(group["path"])
            use_gpu_tm = gpu_available and is_hdr
            hw_args = self._gpu_hw_args(opencl_device) if use_gpu_tm else []

            seek_frame = group["segments"][0][0]
            last_seg = group["segments"][-1]
            last_frame = last_seg[0] + last_seg[1]
            select_expr = self._build_select_expr(group["segments"], seek_frame)

            vf = self._build_vf(
                width, height, use_gpu_tm, output_format="rgb24",
                is_hdr=is_hdr,
                source_width=source.width if source else 0,
                source_height=source.height if source else 0,
                select_expr=select_expr, fps=group["fps"],
            )

            ss = seek_frame / group["fps"] if group["fps"] > 0 else 0
            dur = (last_frame - seek_frame) / group["fps"] if group["fps"] > 0 else 0

            decode_cmd = ["ffmpeg", "-nostdin", "-v", "quiet"] + hw_args + [
                "-hwaccel", "cuda",
                "-ss", f"{ss:.6f}", "-t", f"{dur:.6f}",
                "-i", group["path"],
            ]
            if vf:
                decode_cmd += ["-vf", vf]
            decode_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

            proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            buf = bytearray()
            expected = group["total_frames"]
            frames_read = 0
            while frames_read < expected and not self._cancelled:
                raw = proc.stdout.read(frame_size * 10)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= frame_size and frames_read < expected:
                    frame = np.frombuffer(bytes(buf[:frame_size]),
                                          np.uint8).reshape(height, width, 3)
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
