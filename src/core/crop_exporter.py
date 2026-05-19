"""Standalone exporter for per-clip rectangular crop regions.

Each ``CropRegion`` becomes its own output file (or PNG sequence folder):
- 81 video frames at 16fps (frames *dropped* from source, not slowed)
- Cropped to the rectangle's native source pixels (no resampling)
- Audio is always included, trimmed to the exact 243000-sample window
  matching 81/16 = 5.0625 s at 48 kHz

This is intentionally a separate class from ``core.exporter.Exporter`` —
the per-crop pipeline is a one-shot ffmpeg per crop (no concat), the
filter chain and seek strategy differ from the timeline exporter, and
the file-naming / output-routing semantics are crop-specific. Helpers
that are codec-agnostic (seek timestamps, audio codec map) are imported
from ``Exporter`` for parity.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from core.crop_region import (
    CropRegion, OUTPUT_FPS, OUTPUT_FRAMES, crop_matches_filter,
    exact_audio_samples_81_at_16, required_source_frames,
)
from core.exporter import (
    Exporter, _AUDIO_CODEC_FOR_EXT, _AUDIO_FORMAT_PRESETS,
    _probe_gpu_tonemap,
)
from core.timeline import TimelineModel
from core.video_source import VideoSource
from utils.ffprobe import probe_hdr

logger = logging.getLogger(__name__)


# Synthetic codec ids for the image-sequence paths. Each maps to the
# extension ffmpeg will use for the per-frame filename pattern. Keys
# must match the dialog's image-format combo (`png`, `jpg`, `exr`)
# suffixed with `_sequence`.
IMAGE_SEQUENCE_CODECS: Dict[str, str] = {
    "png_sequence": ".png",
    "jpg_sequence": ".jpg",
    "exr_sequence": ".exr",
}
# Back-compat alias for the originally-shipped PNG codec key.
PNG_SEQUENCE_CODEC = "png_sequence"


def _slug_for_filename(text: str) -> str:
    """Sanitize a group name for use as a folder name. Keeps the
    original characters where safe; falls back to ``_untagged`` for
    empty input."""
    if not text:
        return "_untagged"
    cleaned = re.sub(r"[^A-Za-z0-9 _\-]+", "_", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "_untagged"


def _stem_of(path: str) -> str:
    """Basename without extension. Empty input yields 'source'."""
    if not path:
        return "source"
    return os.path.splitext(os.path.basename(path))[0] or "source"


def _piecewise_n_expr(values) -> str:
    """Build an ffmpeg expression that returns ``values[k]`` when the
    frame counter ``n`` equals ``k``, falling through to the last
    element for any ``n >= len(values)``.

    Output shape:
      ``if(eq(n,0),v0,if(eq(n,1),v1,...,if(eq(n,K-1),v_{K-1},v_{K-1})))``
    Run-length collapses consecutive equal values so a static axis on
    an otherwise-animated region emits a single integer instead of a
    full ladder. Returns the bare value when the entire span is one
    constant.
    """
    if not values:
        return "0"
    # Collapse runs so static axes don't emit a 100-deep ladder.
    runs = []  # list of (start_idx, end_idx_inclusive, value)
    run_start = 0
    for i in range(1, len(values)):
        if values[i] != values[run_start]:
            runs.append((run_start, i - 1, values[run_start]))
            run_start = i
    runs.append((run_start, len(values) - 1, values[run_start]))
    if len(runs) == 1:
        return str(int(runs[0][2]))
    # Build nested if(): one branch per run. We use ``lte(n,end)`` so
    # the value applies from the run's start (covered by the previous
    # branch's failure) through its end. The final run is the fall-
    # through default.
    expr = str(int(runs[-1][2]))
    for start, end, val in reversed(runs[:-1]):
        expr = f"if(lte(n,{end}),{int(val)},{expr})"
    return expr


class CropExporter(QObject):
    """Background-threaded exporter for cropping regions."""

    progress = Signal(int)        # 0-100
    status = Signal(str)
    finished = Signal()
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, timeline: TimelineModel,
                 sources: Dict[str, VideoSource], parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._cancelled = False
        self._active_procs: List[subprocess.Popen] = []
        self._procs_lock = threading.Lock()
        # HDR-probe results, cached per source path for the lifetime of a
        # single export job. Reset at the top of _run so re-exports pick
        # up source remuxes / metadata fixes.
        self._hdr_cache: Dict[str, bool] = {}
        self._hdr_cache_lock = threading.Lock()
        # GPU tonemap state — probed once at the top of _run. When True,
        # HDR sources go through `tonemap_opencl=hable` matching the
        # normal exporter. When False, fall back to the CPU zscale chain.
        # Without this, normal export uses GPU + crop uses CPU and the
        # two algorithms produce visibly different tonemapped output.
        self._gpu_tonemap: bool = False
        self._opencl_device: Optional[str] = None

    # --- Subprocess tracking (mirror Exporter) ------------------------

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

    # --- Entry point --------------------------------------------------

    def export(self, settings: dict):
        """Run the export on a background thread. ``settings`` keys:

        - ``codec`` (str): codec id from ``ui.export_dialog.VIDEO_PRESETS``
          or ``"png_sequence"`` for an image-sequence output
        - ``quality`` (int): replaces ``{quality}`` in the codec args
        - ``output_mode`` (str): ``"root_subfolders"`` or
          ``"per_group_paths"``
        - ``root_dir`` (str): root folder for ``root_subfolders`` mode
        - ``per_group_paths`` (dict): mapping ``group_id`` (or ``None``
          for the Untagged bucket) → folder path. Crops whose group has
          no path mapped are skipped (the dialog warns the user).
        - ``group_filter`` (dict or None): same encoding as
          ``core.group.clip_matches_filter`` — applied at job-build time.
        """
        thread = threading.Thread(
            target=self._run, args=(settings,), daemon=True)
        thread.start()

    # --- Run loop -----------------------------------------------------

    def _run(self, settings: dict):
        try:
            self._cancelled = False
            self._hdr_cache = {}
            self._gpu_tonemap, self._opencl_device = _probe_gpu_tonemap()
            jobs = self._build_jobs(settings)
            if not jobs:
                self.status.emit("No crops to export")
                self.finished.emit()
                return
            self.status.emit(f"Exporting {len(jobs)} crop(s)…")
            self._run_jobs(jobs, settings)
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.progress.emit(100)
                self.status.emit("Crop export complete")
                self.finished.emit()
        except Exception as e:
            logger.exception("Crop export failed")
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(str(e))
                self.status.emit(f"Error: {e}")

    # --- Job planning -------------------------------------------------

    def _build_jobs(self, settings: dict) -> List[dict]:
        """Walk the timeline, collect every active CropRegion that passes
        the group filter, and resolve each one to a fully-specified job."""
        group_filter = settings.get("group_filter")
        output_mode = settings.get("output_mode", "root_subfolders")
        root_dir = settings.get("root_dir") or ""
        per_group_paths: dict = settings.get("per_group_paths") or {}
        codec = settings.get("codec", "h264")

        from ui.export_dialog import VIDEO_PRESETS
        is_image_sequence = (codec in IMAGE_SEQUENCE_CODECS)
        if is_image_sequence:
            preset = None
            out_ext = ""  # not a single file — output is a folder
        else:
            preset = VIDEO_PRESETS.get(codec)
            if preset is None:
                raise ValueError(f"Unknown codec: {codec}")
            out_ext = preset["ext"]

        groups = self._timeline.groups
        jobs: List[dict] = []
        for clip, cr in self._timeline.iter_crops():
            if not cr.active:
                continue
            if not crop_matches_filter(cr, group_filter):
                continue
            source = self._sources.get(clip.source_id)
            if source is None:
                continue
            # Resolve output directory by group.
            group = groups.get(cr.group_id) if cr.group_id else None
            group_name = group.name if group else ""
            target_dir = self._resolve_output_dir(
                output_mode, root_dir, per_group_paths,
                cr.group_id, group_name)
            if target_dir is None:
                # Per-group path not set for this crop's bucket — skip.
                continue
            # Filename: <source>_f{anchor}_{w}x{h}_{crop_id_short}.<ext>
            stem = _stem_of(source.file_path)
            base_name = (
                f"{stem}_f{cr.anchor_frame}_{cr.w}x{cr.h}_{cr.id[:6]}")
            if is_image_sequence:
                # Output is a folder per crop containing frame_001.<ext> ...
                out_path = os.path.join(target_dir, base_name)
            else:
                out_path = os.path.join(target_dir, base_name + out_ext)
            jobs.append({
                "clip_id": clip.id,
                "crop_id": cr.id,
                "crop": cr,
                "source_path": source.file_path,
                "source_fps": source.fps,
                "has_audio": bool(source.audio_channels > 0),
                "out_path": out_path,
                "out_ext": out_ext,
                "is_image_sequence": is_image_sequence,
                "image_ext": IMAGE_SEQUENCE_CODECS.get(codec, ".png"),
                "preset": preset,
            })
        return jobs

    def _resolve_output_dir(self, output_mode: str, root_dir: str,
                            per_group_paths: dict,
                            group_id: Optional[str],
                            group_name: str) -> Optional[str]:
        """Pick the directory a job should land in. Returns ``None`` when
        the per-group path is missing in ``per_group_paths`` mode (those
        crops are skipped)."""
        if output_mode == "per_group_paths":
            path = per_group_paths.get(group_id)
            if not path:
                return None
            os.makedirs(path, exist_ok=True)
            return path
        # Default: root + group-name subfolder. Untagged → "_untagged".
        if not root_dir:
            return None
        sub = _slug_for_filename(group_name or "_untagged")
        target = os.path.join(root_dir, sub)
        os.makedirs(target, exist_ok=True)
        return target

    # --- Worker pool / execution -------------------------------------

    def _run_jobs(self, jobs: List[dict], settings: dict):
        # NVENC: 6 GPU-backed workers max. CPU codecs: scale with cores.
        codec = settings.get("codec", "h264")
        is_nvenc = "nvenc" in codec
        cpu_count = os.cpu_count() or 8
        max_parallel = 6 if is_nvenc else max(2, cpu_count // 4)

        done = 0
        failed = threading.Event()
        total = len(jobs)

        def _run_one(job: dict) -> Tuple[bool, str]:
            if self._cancelled or failed.is_set():
                return (False, "cancelled")
            try:
                self._run_job(job, settings)
                return (True, "")
            except Exception as e:
                logger.exception("Crop export failed: %s", job.get("out_path"))
                return (False, str(e))

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_run_one, j): j for j in jobs}
            for fut in as_completed(futures):
                if self._cancelled:
                    break
                ok, msg = fut.result()
                done += 1
                self.progress.emit(min(99, int(done / max(1, total) * 99)))
                if not ok and not self._cancelled:
                    failed.set()
                    # Kill any other in-flight crops so cancel/error is quick.
                    self.cancel()
                    raise RuntimeError(msg)

    # --- Per-crop pipeline -------------------------------------------

    def _run_job(self, job: dict, settings: dict):
        cr: CropRegion = job["crop"]
        src_path: str = job["source_path"]
        src_fps: float = job["source_fps"]
        has_audio: bool = job["has_audio"]
        out_path: str = job["out_path"]
        is_image_sequence: bool = job["is_image_sequence"]

        if is_image_sequence:
            ext = job.get("image_ext", ".png")
            self._run_image_sequence_job(cr, src_path, src_fps, out_path, ext)
            return

        preset: dict = job["preset"]
        quality = int(settings.get("quality", 18))
        codec_args = self._apply_quality(preset["args"], quality)
        audio_mode = settings.get("audio_mode", "embedded")
        is_hdr = self._is_hdr_source(src_path)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        base, ext = os.path.splitext(out_path)

        if audio_mode == "none":
            # Single-pass video write — no audio track, no mux.
            cmd = self._build_video_cmd(
                cr, src_path, src_fps, codec_args, out_path, is_hdr)
            self._run_pipeline([cmd])
            return

        # embedded (default) or both — three-pass video + audio + mux.
        temp_v = base + ".v" + ext
        temp_a = base + ".a.wav"
        cmds = [
            self._build_video_cmd(
                cr, src_path, src_fps, codec_args, temp_v, is_hdr),
            self._build_audio_cmd(
                cr, src_path, src_fps, has_audio, temp_a),
            self._build_mux_cmd(temp_v, temp_a, out_path),
        ]
        try:
            self._run_pipeline(cmds)
            if audio_mode == "both":
                # Sidecar in the user-chosen format, transcoded from the
                # same sample-precise 243000-sample WAV used for embedding.
                sidecar_cmd = self._build_sidecar_audio_cmd(
                    temp_a, base, settings)
                if sidecar_cmd is not None:
                    self._run_pipeline([sidecar_cmd])
        finally:
            for f in (temp_v, temp_a):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _run_image_sequence_job(self, cr: CropRegion, src_path: str,
                                src_fps: float, out_dir: str,
                                ext: str = ".png"):
        """Write 81 numbered image files (PNG/JPG/EXR) to ``out_dir``.

        Uses the same HDR/color filter chain + two-stage seek as the
        video path so brightness and color match the video output. JPEG
        gets a fixed high-quality ``-q:v 2`` (1=best, 31=worst in ffmpeg).
        PNG and OpenEXR are lossless; no quality knob applies."""
        os.makedirs(out_dir, exist_ok=True)
        is_hdr = self._is_hdr_source(src_path)
        pre_ss, post_ss = self._two_stage_seek(cr.anchor_frame, src_fps)
        cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        if is_hdr:
            cmd += self._gpu_hw_args()
        if pre_ss > 0:
            cmd += ["-ss", f"{pre_ss:.6f}"]
        cmd += ["-i", src_path]
        if post_ss > 0:
            cmd += ["-ss", f"{post_ss:.6f}"]
        cmd += [
            "-an",
            "-vf", self._video_filter(
                cr, is_hdr=is_hdr, include_setpts=False,
                src_fps=src_fps),
            "-frames:v", str(OUTPUT_FRAMES),
            "-fps_mode", "passthrough",
            "-map_chapters", "-1", "-dn", "-sn",
        ]
        # Per-format quality args. PNG / EXR are lossless; JPEG needs a
        # quality value to avoid the default of 31 (worst).
        if ext.lower() in (".jpg", ".jpeg"):
            cmd += ["-q:v", "2"]
        cmd += [os.path.join(out_dir, f"frame_%03d{ext}")]
        self._run_pipeline([cmd])

    # --- Command builders --------------------------------------------

    @staticmethod
    def _animated_crop_filter(cr: CropRegion, src_fps: float) -> str:
        """Build a frame-dependent ``crop=`` filter element when any of
        the region's axis tracks has keyframes. Returns the static
        positional form when the region is fully un-animated.

        The animated form uses ffmpeg's expression evaluator with the
        ``n`` variable (frame index into the crop filter, counting
        source frames after the -ss seek): each axis becomes a 1-deep
        nested ``if(eq(n,k), vk, ...)`` ladder sampled at every source
        frame inside the export window. Per-frame values come from
        ``CropRegion.sample(anchor_frame + k)`` so the expression is a
        snapshot of the keyframe state at export time — once emitted
        the chain runs as plain numeric eval, no per-frame Python.
        """
        if not cr.is_animated():
            return f"crop={cr.w}:{cr.h}:{cr.x}:{cr.y}"
        # Source-frame count to cover the 81-output-frame window. +2
        # frame slack so fps=16 has lookahead room on the last sample.
        n_src = required_source_frames(src_fps) + 2
        anchor = int(cr.anchor_frame)
        xs, ys, ws, hs = [], [], [], []
        for k in range(n_src):
            x, y, w, h = cr.sample(anchor + k)
            xs.append(int(x))
            ys.append(int(y))
            # Width / height must be at least 2 px for the encoder
            # chroma stride. The overlay already enforces MIN_CROP_SIZE
            # but interpolation can land below it briefly.
            ws.append(max(2, int(w)))
            hs.append(max(2, int(h)))
        return (
            "crop="
            f"w='{_piecewise_n_expr(ws)}'"
            f":h='{_piecewise_n_expr(hs)}'"
            f":x='{_piecewise_n_expr(xs)}'"
            f":y='{_piecewise_n_expr(ys)}'"
        )

    def _video_filter(self, cr: CropRegion, is_hdr: bool = False,
                      include_setpts: bool = True,
                      src_fps: float = 0.0) -> str:
        """Filter chain for the cropped 81-frame@16fps window.

        SDR path: ``crop=W:H:X:Y → fps=16 → setpts=PTS-STARTPTS``.

        HDR path: prepended with the same tonemap chain the normal
        exporter uses — GPU `tonemap_opencl=hable` when OpenCL is
        available, CPU `zscale + tonemap=hable + zscale` fallback. The
        leading ``setparams=bt2020/PQ`` forces consistent HDR signalling
        so a mid-GOP seek can't desync the tonemap filter, and the
        trailing ``setparams=bt709`` stamps the output as Rec.709 so
        encoders don't carry forward bt2020 primaries metadata into the
        muxed stream (was visible in pre-fix normal exports as
        `color_primaries=bt2020` despite Rec.709 pixels).

        ``setpts`` rebases to PTS=0 so the muxer writes a clean stream
        starting at t=0; the PNG-sequence path skips it because there's
        no muxer to confuse.
        """
        parts: List[str] = []
        if is_hdr:
            parts.append(
                "setparams=color_primaries=bt2020:"
                "color_trc=smpte2084:colorspace=bt2020nc")
            if self._gpu_tonemap:
                parts.extend([
                    "format=p010le", "hwupload",
                    "tonemap_opencl=tonemap=hable:desat=0:peak=1000:"
                    "format=nv12",
                    "hwdownload", "format=nv12",
                ])
            else:
                parts.extend([
                    "zscale=t=linear:npl=100",
                    "format=gbrpf32le",
                    "zscale=p=bt709",
                    "tonemap=hable:desat=0:peak=10",
                    "zscale=t=bt709:m=bt709:r=tv",
                    "format=yuv420p",
                ])
            parts.append(
                "setparams=color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709")
        parts.append(self._animated_crop_filter(cr, src_fps))
        parts.append(f"fps={OUTPUT_FPS}")
        if include_setpts:
            parts.append("setpts=PTS-STARTPTS")
        return ",".join(parts)

    def _gpu_hw_args(self) -> List[str]:
        """OpenCL init flags for the GPU tonemap path. Empty list when
        GPU tonemap isn't in use; the caller prepends these BEFORE -i."""
        if not (self._gpu_tonemap and self._opencl_device):
            return []
        return ["-init_hw_device", f"opencl=ocl:{self._opencl_device}",
                "-filter_hw_device", "ocl"]

    @staticmethod
    def _two_stage_seek(anchor_frame: int, src_fps: float) -> Tuple[float, float]:
        """Split the seek timestamp into a fast keyframe-accurate pre-input
        seek (~1s before target) + an accurate post-input seek. The
        post-input stage also warms up the filter chain so HDR clips
        don't write their first frame untonemapped (the same fix used in
        the normal exporter)."""
        ss = Exporter._frame_to_seek_ts(anchor_frame, src_fps)
        pre_ss = max(0.0, ss - 1.0)
        post_ss = max(0.0, ss - pre_ss)
        return pre_ss, post_ss

    def _build_video_cmd(self, cr: CropRegion, src_path: str,
                         src_fps: float, codec_args: list,
                         out_file: str, is_hdr: bool = False) -> list:
        pre_ss, post_ss = self._two_stage_seek(cr.anchor_frame, src_fps)
        cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        if is_hdr:
            cmd += self._gpu_hw_args()
        if pre_ss > 0:
            cmd += ["-ss", f"{pre_ss:.6f}"]
        cmd += ["-i", src_path]
        if post_ss > 0:
            cmd += ["-ss", f"{post_ss:.6f}"]
        cmd += [
            "-an",
            "-vf", self._video_filter(cr, is_hdr=is_hdr, src_fps=src_fps),
            "-frames:v", str(OUTPUT_FRAMES),
            "-fps_mode", "passthrough",
            "-video_track_timescale", "24000000",
            # Drop chapter / data / subtitle streams so Blu-ray rips
            # don't bleed their full-runtime chapter tracks into the
            # 5-second crop output.
            "-map_chapters", "-1", "-dn", "-sn",
            # Output color tags — no-op for already-correctly-tagged SDR
            # inputs, load-bearing for HDR inputs after the tonemap chain
            # so players don't reinterpret the bt709 pixels as anything
            # else.
            "-colorspace", "bt709",
            "-color_trc", "bt709",
            "-color_primaries", "bt709",
        ]
        cmd += codec_args + [out_file]
        return cmd

    def _build_sidecar_audio_cmd(self, src_wav: str, out_base: str,
                                 settings: dict) -> Optional[list]:
        """Transcode the embedded WAV to a sidecar file next to the
        video. Returns None if the requested format is unknown — caller
        skips silently rather than failing the whole job."""
        fmt_key = str(settings.get("audio_format", "wav"))
        fmt = _AUDIO_FORMAT_PRESETS.get(fmt_key)
        if not fmt:
            logger.warning("Unknown audio_format %r; skipping sidecar",
                           fmt_key)
            return None
        out_file = out_base + fmt["ext"]
        return [
            "ffmpeg", "-y", "-nostdin", "-v", "error",
            "-i", src_wav,
            "-vn",
            "-c:a", fmt["codec"], *fmt["extra"],
            "-ar", "48000", "-ac", "2",
            out_file,
        ]

    def _is_hdr_source(self, src_path: str) -> bool:
        with self._hdr_cache_lock:
            cached = self._hdr_cache.get(src_path)
        if cached is not None:
            return cached
        try:
            result = bool(probe_hdr(src_path))
        except Exception:
            logger.exception("HDR probe failed for %s", src_path)
            result = False
        with self._hdr_cache_lock:
            self._hdr_cache[src_path] = result
        return result

    def _build_audio_cmd(self, cr: CropRegion, src_path: str,
                         src_fps: float, has_audio: bool,
                         out_wav: str) -> list:
        """Single-stage seek + sample-precise atrim.

        The output rate is the 16fps target (not the source's rate), so
        the sample count is constant (243000 @ 48kHz). We don't reuse
        ``Exporter._exact_audio_samples`` here because that takes source
        frames at source fps — our atrim is sized against output frames
        at 16fps.
        """
        n_samples = exact_audio_samples_81_at_16(48000)
        cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        if has_audio:
            ss = Exporter._frame_to_seek_ts(cr.anchor_frame, src_fps)
            if ss > 0:
                cmd += ["-ss", f"{ss:.6f}"]
            cmd += [
                "-i", src_path,
                "-vn",
                "-map", "0:a:0?",
                "-af",
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"atrim=end_sample={n_samples}",
            ]
        else:
            # Source has no audio — feed in silence sized to the 81/16 window.
            dur = OUTPUT_FRAMES / OUTPUT_FPS
            cmd += [
                "-f", "lavfi", "-t", f"{dur:.6f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                f"-af", f"atrim=end_sample={n_samples}",
            ]
        cmd += ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", out_wav]
        return cmd

    def _build_mux_cmd(self, video_in: str, audio_in: str,
                       out_file: str) -> list:
        out_ext = os.path.splitext(out_file)[1].lower()
        audio_codec, audio_extra = _AUDIO_CODEC_FOR_EXT.get(
            out_ext, ("pcm_s16le", []))
        return [
            "ffmpeg", "-y", "-nostdin", "-v", "error",
            "-i", video_in, "-i", audio_in,
            "-map", "0:v:0", "-map", "1:a:0",
            "-map_chapters", "-1", "-dn", "-sn",
            "-c:v", "copy",
            "-c:a", audio_codec, *audio_extra,
            "-ar", "48000", "-ac", "2",
            "-video_track_timescale", "24000000",
            out_file,
        ]

    # --- Helpers ------------------------------------------------------

    @staticmethod
    def _apply_quality(codec_args: list, quality: int) -> list:
        """Replace ``{quality}`` placeholders in a preset's args."""
        out = []
        for a in codec_args:
            if "{quality}" in a:
                out.append(a.replace("{quality}", str(quality)))
            else:
                out.append(a)
        return out

    def _run_pipeline(self, commands: List[list]):
        """Run ffmpeg commands sequentially. Raises on first failure."""
        for cmd in commands:
            if self._cancelled:
                raise RuntimeError("cancelled")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._register_proc(proc)
            try:
                _, stderr = proc.communicate()
            finally:
                self._unregister_proc(proc)
            if proc.returncode != 0:
                if self._cancelled:
                    raise RuntimeError("cancelled")
                tail = (stderr or b"").decode(errors="replace")[-500:]
                raise RuntimeError(
                    f"ffmpeg failed (rc={proc.returncode}): {tail}")
