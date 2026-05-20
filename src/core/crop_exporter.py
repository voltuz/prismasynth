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
    crop_output_dims, exact_audio_samples_81_at_16, required_source_frames,
    segment_aspect_constant,
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


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from a pipe. Returns None on short read /
    EOF (e.g. when ffmpeg #1 ends before the requested frame count)."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


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
        # Crop labels skipped at job-build time for changing aspect ratio
        # mid-segment (can't be represented at a fixed output resolution).
        # Surfaced in the final status line. Reset at the top of _run.
        self._skipped_ar: List[str] = []

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
        self._kill_active_procs()

    def _kill_active_procs(self):
        """Kill every live subprocess WITHOUT marking the job cancelled.
        Used by the failure path so a real error isn't mis-reported as a
        user cancel (which would otherwise leave the dialog stuck)."""
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
            self._skipped_ar = []
            self._gpu_tonemap, self._opencl_device = _probe_gpu_tonemap()
            jobs = self._build_jobs(settings)
            if not jobs:
                self.status.emit("No crops to export" + self._skip_note())
                self.finished.emit()
                return
            self.status.emit(f"Exporting {len(jobs)} crop(s)…")
            self._run_jobs(jobs, settings)
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.progress.emit(100)
                self.status.emit("Crop export complete" + self._skip_note())
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

        # User-chosen output width (px). Height follows each crop's aspect
        # ratio. 0/missing falls back to the crop's own sampled width.
        out_width = int(settings.get("out_width") or 0)

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
            stem = _stem_of(source.file_path)
            # One job per ACTIVE segment. Filename includes the segment's
            # anchor + crop/segment short ids so windows never collide.
            for seg in cr.segments:
                if not seg.active:
                    continue
                anchor = int(seg.anchor_frame)
                # Crops that change ASPECT RATIO mid-window can't be
                # represented at a fixed output resolution — skip them and
                # record the label so the final status can list them.
                if not segment_aspect_constant(cr, anchor, source.fps):
                    self._note_skipped(cr)
                    continue
                eff_width = out_width if out_width > 0 else cr.sample(anchor)[2]
                out_w, out_h = crop_output_dims(cr, anchor, eff_width)
                base_name = (
                    f"{stem}_f{anchor}_{out_w}x{out_h}"
                    f"_{cr.id[:6]}_{seg.id[:4]}")
                if is_image_sequence:
                    out_path = os.path.join(target_dir, base_name)
                else:
                    out_path = os.path.join(target_dir, base_name + out_ext)
                jobs.append({
                    "clip_id": clip.id,
                    "crop_id": cr.id,
                    "crop": cr,
                    "anchor_frame": anchor,
                    "source_path": source.file_path,
                    "source_fps": source.fps,
                    "source_w": int(source.width),
                    "source_h": int(source.height),
                    "out_w": out_w,
                    "out_h": out_h,
                    "animated": cr.is_animated(),
                    "has_audio": bool(source.audio_channels > 0),
                    "out_path": out_path,
                    "out_ext": out_ext,
                    "is_image_sequence": is_image_sequence,
                    "image_ext": IMAGE_SEQUENCE_CODECS.get(codec, ".png"),
                    "preset": preset,
                })
        return jobs

    def _note_skipped(self, cr: CropRegion):
        label = cr.label.strip() if cr.label else ""
        if not label:
            label = f"crop {cr.id[:6]}"
        if label not in self._skipped_ar:
            self._skipped_ar.append(label)

    def _skip_note(self) -> str:
        if not self._skipped_ar:
            return ""
        names = ", ".join(f"'{n}'" for n in self._skipped_ar)
        n = len(self._skipped_ar)
        return (f" ({n} crop{'s' if n != 1 else ''} skipped — changing "
                f"aspect ratio: {names})")

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
                    # Kill any other in-flight crops so the error surfaces
                    # quickly. Must NOT set self._cancelled — otherwise _run
                    # reports a user-cancel instead of the actual error and
                    # the dialog freezes (was the "stuck at 99%" symptom).
                    self._kill_active_procs()
                    raise RuntimeError(msg)

    # --- Per-crop pipeline -------------------------------------------

    def _run_job(self, job: dict, settings: dict):
        cr: CropRegion = job["crop"]
        anchor: int = int(job["anchor_frame"])
        src_path: str = job["source_path"]
        src_fps: float = job["source_fps"]
        has_audio: bool = job["has_audio"]
        out_path: str = job["out_path"]
        is_image_sequence: bool = job["is_image_sequence"]

        if is_image_sequence:
            ext = job.get("image_ext", ".png")
            self._run_image_sequence_job(job, ext)
            return

        audio_mode = settings.get("audio_mode", "embedded")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        base, ext = os.path.splitext(out_path)

        if audio_mode == "none":
            # No audio track, no mux — write the video straight to out_path.
            self._produce_video(job, settings, out_path)
            return

        # embedded (default) or both — video + audio + mux.
        temp_v = base + ".v" + ext
        temp_a = base + ".a.wav"
        try:
            self._produce_video(job, settings, temp_v)
            self._run_pipeline([
                self._build_audio_cmd(
                    cr, anchor, src_path, src_fps, has_audio, temp_a),
                self._build_mux_cmd(temp_v, temp_a, out_path),
            ])
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

    def _run_image_sequence_job(self, job: dict, ext: str = ".png"):
        """Write 81 numbered image files (PNG/JPG/EXR) to the job's output
        folder, scaled to the job's output dimensions.

        Animated crops go through the per-frame crop+resize pipe; static
        crops use a single ffmpeg pass. Both share the same HDR/color
        chain so brightness and color match the video output. JPEG gets a
        fixed high-quality ``-q:v 2``; PNG / OpenEXR are lossless."""
        cr: CropRegion = job["crop"]
        anchor = int(job["anchor_frame"])
        src_path: str = job["source_path"]
        src_fps: float = job["source_fps"]
        out_dir: str = job["out_path"]
        out_w: int = job["out_w"]
        out_h: int = job["out_h"]
        is_hdr = self._is_hdr_source(src_path)
        os.makedirs(out_dir, exist_ok=True)

        if job["animated"]:
            cmd2 = self._encode_cmd2_images(out_w, out_h, src_fps, out_dir, ext)
            self._render_animated(
                cr, anchor, src_path, src_fps, job["source_w"],
                job["source_h"], out_w, out_h, is_hdr, cmd2)
            return

        pre_ss, post_ss = self._two_stage_seek(anchor, src_fps)
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
                cr, out_w, out_h, is_hdr=is_hdr, include_setpts=False),
            "-frames:v", str(OUTPUT_FRAMES),
            "-fps_mode", "passthrough",
            "-map_chapters", "-1", "-dn", "-sn",
        ]
        if ext.lower() in (".jpg", ".jpeg"):
            cmd += ["-q:v", "2"]
        cmd += [os.path.join(out_dir, f"frame_%03d{ext}")]
        self._run_pipeline([cmd])

    # --- Video production (static pass vs animated per-frame pipe) -----

    def _produce_video(self, job: dict, settings: dict, out_file: str):
        """Write the (audio-less) cropped video to ``out_file``. Animated
        crops render via the per-frame pipe; static crops via one ffmpeg
        pass. Caller handles audio + mux."""
        cr: CropRegion = job["crop"]
        anchor = int(job["anchor_frame"])
        src_path: str = job["source_path"]
        src_fps: float = job["source_fps"]
        out_w: int = job["out_w"]
        out_h: int = job["out_h"]
        is_hdr = self._is_hdr_source(src_path)
        quality = int(settings.get("quality", 18))
        codec_args = self._apply_quality(job["preset"]["args"], quality)
        if job["animated"]:
            cmd2 = self._encode_cmd2_video(
                out_w, out_h, src_fps, codec_args, out_file)
            self._render_animated(
                cr, anchor, src_path, src_fps, job["source_w"],
                job["source_h"], out_w, out_h, is_hdr, cmd2)
        else:
            self._run_pipeline([self._build_video_cmd(
                cr, anchor, src_path, src_fps, codec_args, out_file,
                out_w, out_h, is_hdr)])

    def _render_animated(self, cr: CropRegion, anchor: int, src_path: str,
                         src_fps: float, src_w: int, src_h: int,
                         out_w: int, out_h: int, is_hdr: bool, cmd2: list):
        """ffmpeg #1 (decode + HDR tonemap → bgr24 rawvideo) → Python
        per-frame crop+resize → ffmpeg #2 (``cmd2``: fps=16 + encode, or
        image-sequence write).

        This sidesteps ffmpeg's ``crop`` filter entirely — that filter
        can't animate w/h (no ``eval`` option in this build) and per-frame
        x/y/w/h expressions blow past its evaluator's nesting limit. Doing
        the geometry in Python is exact for arbitrary keyframes. Only 81
        output frames, so the full-res pipe is cheap.
        """
        import numpy as np
        import cv2

        cmd1 = self._decode_cmd1(src_path, src_fps, anchor, is_hdr)
        n_src = required_source_frames(src_fps) + 2
        frame_bytes = src_w * src_h * 3

        p1 = subprocess.Popen(
            cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._register_proc(p1)
        p2 = subprocess.Popen(
            cmd2, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._register_proc(p2)
        err1: List[bytes] = []
        err2: List[bytes] = []
        threading.Thread(
            target=lambda: err1.append(p1.stderr.read()), daemon=True).start()
        threading.Thread(
            target=lambda: err2.append(p2.stderr.read()), daemon=True).start()
        try:
            for n in range(n_src):
                if self._cancelled:
                    break
                buf = _read_exact(p1.stdout, frame_bytes)
                if buf is None:
                    break  # ffmpeg #1 hit EOF or errored
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    src_h, src_w, 3)
                x, y, w, h = cr.sample(anchor + n)
                x = max(0, min(int(x), src_w - 2))
                y = max(0, min(int(y), src_h - 2))
                w = max(2, min(int(w), src_w - x))
                h = max(2, min(int(h), src_h - y))
                sub = frame[y:y + h, x:x + w]
                interp = (cv2.INTER_AREA if (w >= out_w and h >= out_h)
                          else cv2.INTER_CUBIC)
                out = cv2.resize(sub, (out_w, out_h), interpolation=interp)
                try:
                    p2.stdin.write(np.ascontiguousarray(out).tobytes())
                except (BrokenPipeError, OSError):
                    break  # ffmpeg #2 reached its -frames:v cap
        finally:
            try:
                p1.stdout.close()
            except OSError:
                pass
            try:
                p2.stdin.close()
            except OSError:
                pass
        p1.wait()
        p2.wait()
        self._unregister_proc(p1)
        self._unregister_proc(p2)
        if self._cancelled:
            raise RuntimeError("cancelled")
        # ffmpeg #1 returns non-zero when we close its stdout early (EPIPE)
        # after ffmpeg #2 is satisfied — expected, not a failure. Only the
        # encoder's result determines success.
        if p2.returncode != 0:
            t2 = (err2[0] if err2 else b"").decode(errors="replace")[-400:]
            t1 = (err1[0] if err1 else b"").decode(errors="replace")[-400:]
            raise RuntimeError(
                f"crop render failed (rc={p2.returncode}): {t2} | "
                f"decode: {t1}")

    def _decode_cmd1(self, src_path: str, src_fps: float, anchor_frame: int,
                     is_hdr: bool) -> list:
        """ffmpeg #1: two-stage seek + (HDR tonemap) → bgr24 rawvideo on
        stdout, one full source frame per window position. No crop/fps —
        Python crops each frame and ffmpeg #2 drops to 16fps."""
        pre_ss, post_ss = self._two_stage_seek(anchor_frame, src_fps)
        n_src = required_source_frames(src_fps) + 2
        vf = ",".join(self._hdr_prefix_parts(is_hdr) + ["format=bgr24"])
        cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        if is_hdr:
            cmd += self._gpu_hw_args()
        if pre_ss > 0:
            cmd += ["-ss", f"{pre_ss:.6f}"]
        cmd += ["-i", src_path]
        if post_ss > 0:
            cmd += ["-ss", f"{post_ss:.6f}"]
        cmd += ["-an", "-vf", vf, "-frames:v", str(n_src),
                "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
        return cmd

    def _encode_cmd2_video(self, out_w: int, out_h: int, src_fps: float,
                           codec_args: list, out_file: str) -> list:
        """ffmpeg #2 for video: bgr24 rawvideo on stdin → fps=16 → encode.
        No ``-nostdin`` — this process reads frames from stdin."""
        return [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{src_fps:.6f}", "-i", "-",
            "-vf", f"fps={OUTPUT_FPS},setpts=PTS-STARTPTS",
            "-frames:v", str(OUTPUT_FRAMES), "-fps_mode", "passthrough",
            "-video_track_timescale", "24000000",
            "-map_chapters", "-1", "-dn", "-sn",
            "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709",
            *codec_args, out_file,
        ]

    def _encode_cmd2_images(self, out_w: int, out_h: int, src_fps: float,
                            out_dir: str, ext: str) -> list:
        """ffmpeg #2 for image sequences: bgr24 rawvideo on stdin → fps=16
        → numbered image files. No ``-nostdin`` (reads stdin)."""
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{src_fps:.6f}", "-i", "-",
            "-vf", f"fps={OUTPUT_FPS}",
            "-frames:v", str(OUTPUT_FRAMES), "-fps_mode", "passthrough",
            "-map_chapters", "-1", "-dn", "-sn",
        ]
        if ext.lower() in (".jpg", ".jpeg"):
            cmd += ["-q:v", "2"]
        cmd += [os.path.join(out_dir, f"frame_%03d{ext}")]
        return cmd

    # --- Command builders --------------------------------------------

    def _hdr_prefix_parts(self, is_hdr: bool) -> List[str]:
        """The HDR→Rec.709 tonemap filter elements (GPU ``tonemap_opencl``
        when OpenCL is available, CPU ``zscale``/``tonemap`` fallback
        otherwise), bracketed by ``setparams``. Empty for SDR sources.

        The leading ``setparams=bt2020/PQ`` forces consistent HDR
        signalling so a mid-GOP seek can't desync the tonemap filter; the
        trailing ``setparams=bt709`` stamps the result Rec.709 so a
        downstream encoder doesn't carry bt2020 primaries forward."""
        if not is_hdr:
            return []
        parts = ["setparams=color_primaries=bt2020:"
                 "color_trc=smpte2084:colorspace=bt2020nc"]
        if self._gpu_tonemap:
            parts += [
                "format=p010le", "hwupload",
                "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12",
                "hwdownload", "format=nv12",
            ]
        else:
            parts += [
                "zscale=t=linear:npl=100", "format=gbrpf32le",
                "zscale=p=bt709", "tonemap=hable:desat=0:peak=10",
                "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p",
            ]
        parts.append("setparams=color_primaries=bt709:"
                     "color_trc=bt709:colorspace=bt709")
        return parts

    def _video_filter(self, cr: CropRegion, out_w: int, out_h: int,
                      is_hdr: bool = False,
                      include_setpts: bool = True) -> str:
        """Filter chain for a STATIC (non-animated) crop:
        ``[hdr tonemap] → crop=W:H:X:Y → scale=out_w:out_h → fps=16
        [→ setpts]``.

        Animated crops never reach here — they go through the per-frame
        Python pipe (``_render_animated``). ``setpts`` rebases to PTS=0
        for clean muxing; the image-sequence path skips it (no muxer)."""
        parts = self._hdr_prefix_parts(is_hdr)
        parts.append(f"crop={int(cr.w)}:{int(cr.h)}:{int(cr.x)}:{int(cr.y)}")
        parts.append(f"scale={out_w}:{out_h}")
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

    def _build_video_cmd(self, cr: CropRegion, anchor_frame: int,
                         src_path: str, src_fps: float, codec_args: list,
                         out_file: str, out_w: int, out_h: int,
                         is_hdr: bool = False) -> list:
        pre_ss, post_ss = self._two_stage_seek(anchor_frame, src_fps)
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
            "-vf", self._video_filter(cr, out_w, out_h, is_hdr=is_hdr),
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

    def _build_audio_cmd(self, cr: CropRegion, anchor_frame: int,
                         src_path: str, src_fps: float, has_audio: bool,
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
            ss = Exporter._frame_to_seek_ts(anchor_frame, src_fps)
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
