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

# Per-segment audio is re-encoded to a uniform codec/rate/channels so the
# downstream `-c copy` concat step survives across mixed sources. PCM for
# ProRes/FFV1 containers, AAC for MP4. Map keyed on output extension.
_AUDIO_CODEC_FOR_EXT = {
    ".mov": ("pcm_s16le", []),
    ".mkv": ("pcm_s16le", []),
    ".mp4": ("aac", ["-b:a", "320k"]),
}

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
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, timeline: TimelineModel, sources: Dict[str, VideoSource],
                 parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._cancelled = False
        self._active_procs: List[subprocess.Popen] = []
        self._procs_lock = threading.Lock()

    def _register_proc(self, proc: subprocess.Popen):
        with self._procs_lock:
            self._active_procs.append(proc)

    def _unregister_proc(self, proc: subprocess.Popen):
        with self._procs_lock:
            if proc in self._active_procs:
                self._active_procs.remove(proc)

    def cancel(self):
        self._cancelled = True
        with self._procs_lock:
            procs = list(self._active_procs)
            self._active_procs.clear()
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
    def _frame_to_seek_ts(frame: int, fps: float) -> float:
        """Convert a source frame number to a safe ffmpeg seek timestamp.

        Using `frame / fps` directly can round microscopically above the
        target frame's true PTS (because fps is an IEEE 754 approximation
        of e.g. 24000/1001). ffmpeg's accurate-seek then drops that frame
        and emits frame+1, producing an off-by-one segment. Subtracting
        half a frame forces the seek target mid-way between (frame-1)
        and frame, guaranteeing `frame` is the first frame kept.
        """
        if frame <= 0 or fps <= 0:
            return 0.0
        return (frame - 0.5) / fps

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
            # setparams forces HDR10/BT.2020 metadata on every frame so
            # tonemap_opencl has consistent input signalling even when
            # the seek lands on a mid-GOP P/B frame whose per-frame
            # metadata may be missing — otherwise the first frame of
            # that clip passes through the tonemap uncorrected.
            parts.append(
                "setparams=color_primaries=bt2020:"
                "color_trc=smpte2084:colorspace=bt2020nc")
            # Explicit peak on the tonemap filter gives a static, movie-wide
            # curve derived from the stream's declared color space (PQ/BT.2020),
            # instead of per-segment auto-detection from frame samples. This
            # avoids the "tonemap ramps in" look on each clip's first frames
            # and ignores DV's per-scene dynamic metadata. 1000 nits is the
            # HDR10 reference peak; tonemap_opencl takes peak in cd/m² while
            # the CPU tonemap filter takes it relative to npl (here 100),
            # so 1000 nits → peak=10 in that chain.
            if gpu_tonemap:
                parts.extend([
                    "format=p010le", "hwupload",
                    "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12",
                    "hwdownload", "format=nv12",
                ])
            else:
                parts.extend([
                    "zscale=t=linear:npl=100", "format=gbrpf32le",
                    "zscale=p=bt709", "tonemap=hable:desat=0:peak=10",
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
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit()
        except Exception as e:
            logger.exception("Export failed")
            if self._cancelled:
                self.cancelled.emit()
            else:
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

        # Flatten coalesced segments back out with per-source VF chains.
        # Cache HDR detection on the group dict so probe_hdr runs once per
        # source rather than 2-3x per export.
        flat_segments = []  # [(source_path, src_in, count, fps, vf, hw_args, is_hdr, src_w, src_h, has_audio)]
        for sid, group in groups.items():
            source = self._sources.get(sid)
            group["is_hdr"] = probe_hdr(group["path"])
            is_hdr = group["is_hdr"]
            use_gpu_tm = gpu_available and is_hdr
            hw = self._gpu_hw_args(opencl_device) if use_gpu_tm else []
            src_w = source.width if source else 0
            src_h = source.height if source else 0
            has_audio = bool(source and source.audio_channels > 0)
            # For SDR + NVENC we keep the entire pipeline on the GPU
            # (NVDEC -> scale_cuda -> NVENC, no GPU<->CPU roundtrip).
            # _build_segment_cmd handles the scale_cuda filter when vf is
            # None, sized from src_w/src_h vs target width/height.
            if is_nvenc and not is_hdr:
                vf = None
            else:
                vf = self._build_vf(
                    width, height, use_gpu_tm, is_hdr=is_hdr,
                    source_width=src_w, source_height=src_h,
                )
            for src_in, count in group["segments"]:
                flat_segments.append(
                    (group["path"], src_in, count, group["fps"], vf, hw,
                     is_hdr, src_w, src_h, has_audio))

        total_frames = sum(s[2] for s in flat_segments)
        suffix = ""
        if any(g["is_hdr"] for g in groups.values()):
            suffix = " (GPU tonemap)" if gpu_available else " (CPU tonemap)"
        self.status.emit(
            f"Encoding {len(flat_segments)} segments to {output_path}...{suffix}")

        def _build_segment_cmd(path, src_in, count, src_fps, vf, hw, hdr,
                               src_w, src_h, has_audio, out_file):
            """Build FFmpeg command for a single segment."""
            ss = self._frame_to_seek_ts(src_in, src_fps)
            hwaccel = ["-hwaccel", "cuda"]
            # SDR + NVENC: full GPU pipeline (zero-copy). scale_cuda
            # handles both format conversion to NVENC-compatible yuv420p
            # and any resolution change (e.g. 4K->1080p) in one CUDA kernel,
            # eliminating a GPU->CPU->GPU roundtrip.
            if not vf and is_nvenc:
                hwaccel += ["-hwaccel_output_format", "cuda"]
                need_scale = (src_w > 0 and src_h > 0
                              and (src_w != width or src_h != height))
                if need_scale:
                    gpu_vf = f"scale_cuda={width}:{height}:format=yuv420p"
                else:
                    gpu_vf = "scale_cuda=format=yuv420p"
            else:
                gpu_vf = None
            # Two-stage seek: pre-input -ss jumps to ~1s before target
            # (keyframe-accurate, cheap); post-input -ss accurately drops
            # the final second. The post-input stage is the one that
            # matters — it pushes ~24 frames through the filter graph
            # BEFORE the first output frame, which warms up the OpenCL
            # tonemap context. Without that warmup the first output frame
            # of some clips bypasses the tonemap (raw HDR colors).
            PRE_SEEK_MARGIN = 1.0
            if ss > PRE_SEEK_MARGIN:
                pre_ss, post_ss = ss - PRE_SEEK_MARGIN, PRE_SEEK_MARGIN
            else:
                pre_ss, post_ss = 0.0, ss
            cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"] + hw + hwaccel
            # Source has no audio: feed in a silent stereo lavfi input sized
            # to the segment's video duration. It MUST be added BEFORE the
            # video input so that the post-input -ss below stays in
            # output-option position (no -i follows it). Putting anullsrc
            # AFTER the video would cause ffmpeg to reinterpret the post-
            # input -ss as input options for anullsrc, breaking the video's
            # accurate-seek and the OpenCL tonemap warmup. Maps below flip
            # accordingly: video=input1:v, silent-audio=input0:a.
            # The lavfi -t includes post_ss because output -ss drops post_ss
            # seconds from ALL output streams (audio included), so the
            # anullsrc must over-produce by post_ss to leave audio_dur after.
            audio_dur = count / src_fps
            if not has_audio:
                cmd += ["-f", "lavfi", "-t", f"{audio_dur + post_ss:.6f}",
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            video_idx = 1 if not has_audio else 0
            if pre_ss > 0:
                cmd += ["-ss", f"{pre_ss:.6f}"]
            cmd += ["-i", path]
            if post_ss > 0:
                cmd += ["-ss", f"{post_ss:.6f}"]
            cmd += ["-frames:v", str(count)]
            if vf:
                # setpts=PTS-STARTPTS rebases each segment's first frame to
                # PTS=0 so the stream-copy concat can append them cleanly.
                cmd += ["-vf", f"setpts=PTS-STARTPTS,{vf}"]
            elif gpu_vf:
                cmd += ["-vf", gpu_vf]
            if hdr:
                cmd += ["-colorspace", "bt709", "-color_trc", "bt709",
                        "-color_primaries", "bt709"]
            if not is_nvenc:
                threads_per = max(2, cpu_count // max_parallel)
                cmd += ["-threads", str(threads_per)]
            # -fps_mode passthrough: required — without it, some clips'
            # first output frame bypasses the tonemap filter (filter-graph
            # initialization race, likely CFR mode emitting before the
            # OpenCL pipeline is ready).
            # -video_track_timescale 24000000 represents 23.976 (24000/1001),
            # 29.97 (30000/1001), 24, 30, 60 etc. as exact-integer frame
            # durations in the MOV/MP4 muxer. Without it the default 1/16000
            # timebase stores jittering durations (672, 656, ...) which NLEs
            # interpret strictly and display as duplicate frames.
            out_ext = os.path.splitext(out_file)[1].lower()
            audio_codec, audio_extra = _AUDIO_CODEC_FOR_EXT.get(
                out_ext, ("pcm_s16le", []))
            cmd += ["-map", f"{video_idx}:v:0"]
            if has_audio:
                # asetpts=PTS-STARTPTS is the audio analogue of setpts above.
                # atrim bounds the audio to the same window as -frames:v so
                # decoder-tail packets can't bleed past the video's last frame.
                # The atrim runs INSIDE the -af filter graph, which executes
                # BEFORE the output-side -ss discard, so we must add post_ss
                # to the duration — the trim then keeps (audio_dur + post_ss)
                # seconds, output -ss drops the first post_ss, leaving exactly
                # audio_dur to match the video. Without this compensation,
                # audio would be (audio_dur - post_ss) — a 1s shortfall.
                # aresample=async=1:first_pts=0 flushes leading samples that
                # fell before the seek point and resets PTS to 0.
                trim_dur = audio_dur + post_ss
                cmd += [
                    "-map", f"{video_idx}:a:0?",
                    "-af",
                    "asetpts=PTS-STARTPTS,"
                    f"atrim=duration={trim_dur:.6f},"
                    "aresample=async=1:first_pts=0,"
                    "aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo",
                ]
            else:
                cmd += ["-map", "0:a:0"]
            cmd += ["-c:a", audio_codec, *audio_extra,
                    "-ar", "48000", "-ac", "2"]
            cmd += ["-fps_mode", "passthrough",
                    "-video_track_timescale", "24000000"]
            cmd += ffmpeg_args + [out_file]
            return cmd

        # Single segment: encode directly to output (no temp files, no concat)
        if len(flat_segments) == 1:
            (path, src_in, count, src_fps, vf, hw, hdr,
             src_w, src_h, has_audio) = flat_segments[0]
            cmd = _build_segment_cmd(path, src_in, count, src_fps, vf, hw,
                                     hdr, src_w, src_h, has_audio, output_path)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._register_proc(proc)
            try:
                _, stderr_data = proc.communicate()
            finally:
                self._unregister_proc(proc)
            if proc.returncode != 0 and not self._cancelled:
                err = (stderr_data or b"").decode(errors="replace")[-500:]
                raise RuntimeError(f"Encoding failed: {err}")
            self.progress.emit(100)
            self.status.emit(f"Video saved to {output_path}")
            return

        # Multiple segments: temp files + parallel encode + concat
        # abspath first so a bare filename doesn't yield dirname="" (mkdtemp crash).
        # Temp extension matches the output so muxer-specific options
        # (e.g. -video_track_timescale for MOV/MP4) take effect per-segment
        # and are preserved by the stream-copy concat.
        temp_dir = tempfile.mkdtemp(
            prefix="prismasynth_export_",
            dir=os.path.dirname(os.path.abspath(output_path)))
        temp_ext = os.path.splitext(output_path)[1] or ".mkv"
        try:
            temp_files = []
            commands = []
            for i, (path, src_in, count, src_fps, vf, hw, hdr,
                    src_w, src_h, has_audio) in enumerate(flat_segments):
                tmp = os.path.join(temp_dir, f"seg_{i:04d}{temp_ext}")
                temp_files.append(tmp)
                cmd = _build_segment_cmd(path, src_in, count, src_fps, vf, hw,
                                         hdr, src_w, src_h, has_audio, tmp)
                commands.append((cmd, count))

            # Thread pool: as-soon-as-done scheduling (no batch-wait)
            frames_done = 0
            failed = threading.Event()

            def _encode_segment(idx):
                if self._cancelled or failed.is_set():
                    return idx, -1, b"cancelled"
                cmd, _ = commands[idx]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                self._register_proc(proc)
                try:
                    _, stderr_data = proc.communicate()
                finally:
                    self._unregister_proc(proc)
                return idx, proc.returncode, stderr_data or b""

            def _kill_active():
                with self._procs_lock:
                    procs = list(self._active_procs)
                for p in procs:
                    try:
                        p.kill()
                    except OSError:
                        pass

            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                futures = {pool.submit(_encode_segment, i): i
                           for i in range(len(commands))}
                for future in as_completed(futures):
                    if self._cancelled:
                        return  # cancel() already killed procs
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

            if self._cancelled:
                return

            # Concat
            self.status.emit("Concatenating segments...")
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for tmp in temp_files:
                    escaped = tmp.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            # -video_track_timescale applies to the final MOV/MP4 muxer;
            # keep it on concat (not per-segment) because temp files are
            # .mkv which would ignore it.
            concat_cmd = [
                "ffmpeg", "-y", "-nostdin", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c", "copy",
                "-video_track_timescale", "24000000",
                output_path,
            ]
            result = subprocess.run(concat_cmd, capture_output=True)
            if result.returncode != 0 and not self._cancelled:
                stderr = result.stderr.decode(errors="replace")[-500:]
                raise RuntimeError(f"Concat failed: {stderr}")

            self.progress.emit(100)
            self.status.emit(f"Video saved to {output_path}")
        finally:
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

        # abspath first so a bare filename doesn't yield dirname="" (mkdtemp crash).
        temp_dir = tempfile.mkdtemp(
            prefix="prismasynth_export_",
            dir=os.path.dirname(os.path.abspath(output_path)))
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
        # Temp extension matches output (see _export_video_parallel comment).
        temp_ext = os.path.splitext(output_path)[1] or ".mkv"

        audio_codec, audio_extra = _AUDIO_CODEC_FOR_EXT.get(
            temp_ext.lower(), ("pcm_s16le", []))

        for i, (source_path, src_in, duration, src_fps, _sid) in enumerate(segments):
            ss = self._frame_to_seek_ts(src_in, src_fps)
            tmp = os.path.join(temp_dir, f"seg_{i:04d}{temp_ext}")
            temp_files.append(tmp)
            src = self._sources.get(_sid)
            has_audio = bool(src and src.audio_channels > 0)
            audio_dur = duration / src_fps

            # Two-stage seek — see _build_segment_cmd above for rationale.
            PRE_SEEK_MARGIN = 1.0
            if ss > PRE_SEEK_MARGIN:
                pre_ss, post_ss = ss - PRE_SEEK_MARGIN, PRE_SEEK_MARGIN
            else:
                pre_ss, post_ss = 0.0, ss
            vf_chain = f"setpts=PTS-STARTPTS,{vf}" if vf else None
            cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"] + hw_args + [
                "-hwaccel", "cuda",
            ]
            # See _build_segment_cmd for why anullsrc must be input #0 and
            # why -t must include post_ss.
            if not has_audio:
                cmd += ["-f", "lavfi", "-t", f"{audio_dur + post_ss:.6f}",
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            video_idx = 1 if not has_audio else 0
            if pre_ss > 0:
                cmd += ["-ss", f"{pre_ss:.6f}"]
            cmd += ["-i", source_path]
            if post_ss > 0:
                cmd += ["-ss", f"{post_ss:.6f}"]
            cmd += ["-frames:v", str(duration)]
            if vf_chain:
                cmd += ["-vf", vf_chain]
            cmd += ["-map", f"{video_idx}:v:0"]
            if has_audio:
                trim_dur = audio_dur + post_ss
                cmd += [
                    "-map", f"{video_idx}:a:0?",
                    "-af",
                    "asetpts=PTS-STARTPTS,"
                    f"atrim=duration={trim_dur:.6f},"
                    "aresample=async=1:first_pts=0,"
                    "aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo",
                ]
            else:
                cmd += ["-map", "0:a:0"]
            cmd += ["-c:a", audio_codec, *audio_extra,
                    "-ar", "48000", "-ac", "2"]
            cmd += ["-fps_mode", "passthrough",
                    "-video_track_timescale", "24000000"]
            cmd += ffmpeg_args + [tmp]
            commands.append((cmd, duration))

        # Execute in batches
        frames_done = 0
        for batch_start in range(0, len(commands), max_parallel):
            if self._cancelled:
                return
            batch = commands[batch_start:batch_start + max_parallel]
            procs = []
            for cmd, _ in batch:
                p = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                procs.append(p)
                self._register_proc(p)

            # Drain stderr concurrently. Previously this path used
            # p.wait() + p.stderr.read() after — which deadlocks when ffmpeg
            # fills the stderr pipe buffer (~64KB) before terminating.
            stderr_data = [b""] * len(procs)

            def _drain(idx: int, proc: subprocess.Popen):
                try:
                    _, err = proc.communicate()
                    stderr_data[idx] = err or b""
                except Exception:
                    stderr_data[idx] = b""

            threads = [
                threading.Thread(target=_drain, args=(i, p), daemon=True)
                for i, p in enumerate(procs)
            ]
            for t in threads:
                t.start()

            while any(t.is_alive() for t in threads):
                if self._cancelled:
                    for pp in procs:
                        try:
                            pp.kill()
                        except OSError:
                            pass
                    for t in threads:
                        t.join(timeout=2.0)
                    for pp in procs:
                        self._unregister_proc(pp)
                    return
                for t in threads:
                    t.join(timeout=0.25)

            for p in procs:
                self._unregister_proc(p)

            # Check for segment failures
            for idx_in_batch, p in enumerate(procs):
                if p.returncode != 0:
                    seg_idx = batch_start + idx_in_batch
                    err = stderr_data[idx_in_batch].decode(
                        errors="replace")[-500:]
                    logger.error("Segment %d failed (code %d): %s",
                                 seg_idx, p.returncode, err)
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
            "-c", "copy",
            "-video_track_timescale", "24000000",
            output_path,
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

            ss = self._frame_to_seek_ts(seek_frame, group["fps"])
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
            self._register_proc(proc)
            try:
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
            finally:
                # Always clean up, even if the consumer raised or we broke
                # out early. Kill first so proc.wait() returns promptly;
                # closing stdout on its own can hang when ffmpeg is still
                # writing frames.
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                try:
                    proc.stdout.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                self._unregister_proc(proc)

    @staticmethod
    def _save_exr(frame_rgb: np.ndarray, path: str):
        try:
            import imageio
            frame_float = frame_rgb.astype(np.float32) / 255.0
            imageio.imwrite(path, frame_float)
        except ImportError:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.replace(".exr", ".png"), bgr)
