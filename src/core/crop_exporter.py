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
    exact_audio_samples_81_at_16,
)
from core.exporter import Exporter, _AUDIO_CODEC_FOR_EXT
from core.timeline import TimelineModel
from core.video_source import VideoSource

logger = logging.getLogger(__name__)


# Synthetic codec id for the image-sequence path (no entry in VIDEO_PRESETS).
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
        is_png_sequence = (codec == PNG_SEQUENCE_CODEC)
        if not is_png_sequence:
            preset = VIDEO_PRESETS.get(codec)
            if preset is None:
                raise ValueError(f"Unknown codec: {codec}")
            out_ext = preset["ext"]
        else:
            preset = None
            out_ext = ""  # not a single file

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
            if is_png_sequence:
                # Output is a folder per crop containing frame_001.png ...
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
                "is_png_sequence": is_png_sequence,
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
        is_png: bool = job["is_png_sequence"]

        if is_png:
            self._run_png_sequence_job(cr, src_path, src_fps, out_path)
            return

        preset: dict = job["preset"]
        quality = int(settings.get("quality", 18))
        codec_args = self._apply_quality(preset["args"], quality)
        out_ext = job["out_ext"]

        # Video + audio + mux pipeline (audio always included per spec).
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        base, ext = os.path.splitext(out_path)
        temp_v = base + ".v" + ext
        temp_a = base + ".a.wav"

        cmds = [
            self._build_video_cmd(cr, src_path, src_fps, codec_args, temp_v),
            self._build_audio_cmd(cr, src_path, src_fps, has_audio, temp_a),
            self._build_mux_cmd(temp_v, temp_a, out_path),
        ]
        try:
            self._run_pipeline(cmds)
        finally:
            for f in (temp_v, temp_a):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _run_png_sequence_job(self, cr: CropRegion, src_path: str,
                              src_fps: float, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        ss = Exporter._frame_to_seek_ts(cr.anchor_frame, src_fps)
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-v", "error",
        ]
        if ss > 0:
            cmd += ["-ss", f"{ss:.6f}"]
        cmd += [
            "-i", src_path,
            "-an",
            "-vf", self._video_filter(cr, include_setpts=False),
            "-frames:v", str(OUTPUT_FRAMES),
            "-fps_mode", "passthrough",
            os.path.join(out_dir, "frame_%03d.png"),
        ]
        self._run_pipeline([cmd])

    # --- Command builders --------------------------------------------

    def _video_filter(self, cr: CropRegion,
                      include_setpts: bool = True) -> str:
        """``crop=W:H:X:Y`` → ``fps=16`` → (``setpts=PTS-STARTPTS``).

        ``setpts`` rebases to PTS=0 so the muxer writes a clean stream
        starting at t=0; the PNG-sequence path skips it because there's
        no muxer to confuse.
        """
        parts = [f"crop={cr.w}:{cr.h}:{cr.x}:{cr.y}",
                 f"fps={OUTPUT_FPS}"]
        if include_setpts:
            parts.append("setpts=PTS-STARTPTS")
        return ",".join(parts)

    def _build_video_cmd(self, cr: CropRegion, src_path: str,
                         src_fps: float, codec_args: list,
                         out_file: str) -> list:
        ss = Exporter._frame_to_seek_ts(cr.anchor_frame, src_fps)
        cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        if ss > 0:
            cmd += ["-ss", f"{ss:.6f}"]
        cmd += [
            "-i", src_path,
            "-an",
            "-vf", self._video_filter(cr),
            "-frames:v", str(OUTPUT_FRAMES),
            "-fps_mode", "passthrough",
            "-video_track_timescale", "24000000",
        ]
        cmd += codec_args + [out_file]
        return cmd

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
