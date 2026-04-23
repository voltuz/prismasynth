"""PrismaSynth systems + performance benchmark.

Run:  venv\\Scripts\\python scripts\\benchmark.py [--video PATH] [--video PATH ...]
             [--duration SECS] [--skip SECTION ...]

Without --video the video-dependent sections (probe, thumbnail decode,
scene-detection decode, export) are skipped. Sections: env, compile, model,
probe, thumb, decode, export.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import py_compile
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# libmpv-2.dll lives in src/ — the real app's PreviewWidget prepends this to
# PATH before `import mpv`. Replicate that here so the mpv import doesn't
# fail at module load.
os.environ["PATH"] = str(SRC) + os.pathsep + os.environ.get("PATH", "")

# Quiet noisy debug logs from subsystems under test
logging.basicConfig(level=logging.WARNING)

SECTIONS = ("env", "compile", "model", "probe", "thumb", "decode", "export")


# ----------------------------- output helpers -----------------------------

class Colors:
    G = "\033[32m"   # green
    Y = "\033[33m"   # yellow
    R = "\033[31m"   # red
    C = "\033[36m"   # cyan
    B = "\033[1m"    # bold
    X = "\033[0m"    # reset


def _enable_win_ansi():
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        k.GetConsoleMode(h, ctypes.byref(mode))
        k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


def header(text: str):
    bar = "=" * 70
    print(f"\n{Colors.B}{Colors.C}{bar}\n{text}\n{bar}{Colors.X}")


def subheader(text: str):
    print(f"\n{Colors.B}-- {text} --{Colors.X}")


def ok(msg: str):
    print(f"  {Colors.G}[OK]{Colors.X} {msg}")


def warn(msg: str):
    print(f"  {Colors.Y}[WARN]{Colors.X} {msg}")


def fail(msg: str):
    print(f"  {Colors.R}[FAIL]{Colors.X} {msg}")


def info(msg: str):
    print(f"  {msg}")


def fmt_ms(seconds: float) -> str:
    if seconds < 0:
        return "     -"
    if seconds < 1.0:
        return f"{seconds * 1000:6.1f} ms"
    return f"{seconds:6.2f}  s"


# ----------------------------- result tracking ----------------------------

@dataclass
class SectionResult:
    name: str
    passed: bool
    warnings: int = 0
    timings: dict | None = None
    note: str = ""


RESULTS: List[SectionResult] = []


def run_section(name: str, fn: Callable[[], SectionResult]) -> SectionResult:
    try:
        res = fn()
    except Exception:
        traceback.print_exc()
        res = SectionResult(name=name, passed=False, note="exception — see trace")
    RESULTS.append(res)
    return res


# --------------------------- section 1: env -------------------------------

def run_version(cmd: list, pattern: str = None) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        text = (out.stdout or "") + (out.stderr or "")
        if pattern:
            import re
            m = re.search(pattern, text)
            if m:
                return m.group(0)
        return text.splitlines()[0] if text else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def section_env() -> SectionResult:
    header("Environment check")
    warnings = 0

    info(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    info(f"PYTHON_GIL={os.environ.get('PYTHON_GIL', '(unset)')}")

    ffmpeg = run_version(["ffmpeg", "-version"], r"ffmpeg version \S+")
    if ffmpeg:
        ok(f"FFmpeg: {ffmpeg}")
    else:
        fail("FFmpeg not on PATH")
        return SectionResult("env", passed=False, note="no FFmpeg")

    ffprobe = run_version(["ffprobe", "-version"], r"ffprobe version \S+")
    if ffprobe:
        ok(f"ffprobe: {ffprobe}")
    else:
        fail("ffprobe not on PATH")
        return SectionResult("env", passed=False, note="no ffprobe")

    libmpv = SRC / "libmpv-2.dll"
    if libmpv.exists():
        ok(f"libmpv-2.dll: present ({libmpv.stat().st_size // 1024} KB)")
    else:
        warn(f"libmpv-2.dll not found at {libmpv}")
        warnings += 1

    # NVENC available? Must scan the whole -encoders output, not just the
    # first line (which is always the version banner).
    try:
        enc_out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        enc_out = ""
    has_h264_nvenc = "h264_nvenc" in enc_out
    has_hevc_nvenc = "hevc_nvenc" in enc_out
    if has_h264_nvenc and has_hevc_nvenc:
        ok("NVENC encoders: h264_nvenc, hevc_nvenc")
    else:
        missing = [n for n, present in
                   (("h264_nvenc", has_h264_nvenc),
                    ("hevc_nvenc", has_hevc_nvenc)) if not present]
        warn(f"NVENC encoders missing: {', '.join(missing) or '?'}")
        warnings += 1

    # Python packages
    for mod, label in [("PySide6", "PySide6"), ("mpv", "python-mpv"),
                       ("av", "PyAV"), ("cv2", "opencv-python"),
                       ("numpy", "numpy"), ("torch", "torch")]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{label}: {ver}")
        except ImportError:
            fail(f"{label}: not installed")
            warnings += 1
        except OSError as e:
            # python-mpv raises OSError if libmpv-2.dll isn't findable
            fail(f"{label}: failed to load — {e}")
            warnings += 1

    # CUDA for scene detection + NVENC
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            ok(f"CUDA: {torch.version.cuda}, device 0 = {name}")
        else:
            warn("CUDA not available — scene detection will fall back to CPU")
            warnings += 1
    except ImportError:
        pass

    return SectionResult("env", passed=True, warnings=warnings)


# ------------------------ section 2: compile ------------------------------

def section_compile() -> SectionResult:
    header("Module compilation")
    failed = []
    py_files = sorted(SRC.rglob("*.py"))
    for f in py_files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append((f.relative_to(ROOT), str(e)))
    if failed:
        for path, err in failed:
            fail(f"{path}: {err}")
        return SectionResult("compile", passed=False, note=f"{len(failed)} file(s) failed")
    ok(f"{len(py_files)} file(s) compile cleanly")
    return SectionResult("compile", passed=True)


# ------------------------ section 3: data model ---------------------------

def section_model() -> SectionResult:
    header("Data model synthetic benchmark")
    from core.clip import Clip
    from core.timeline import TimelineModel
    from core.project import save_project, load_project

    # Need a QCoreApplication because TimelineModel is a QObject
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    tl = TimelineModel()
    N = 2000
    src_id = "bench_src"
    clips = [Clip(source_id=src_id, source_in=i * 48, source_out=i * 48 + 47)
             for i in range(N)]

    t0 = time.perf_counter()
    tl.add_clips(clips, assign_colors=False)
    t_add = time.perf_counter() - t0
    ok(f"add_clips (N={N}): {fmt_ms(t_add)}")

    # split every 3rd clip
    to_split = [tl.clips[i].id for i in range(0, N, 3)]
    t0 = time.perf_counter()
    for cid in to_split:
        tl_start = tl.get_clip_timeline_start(cid)
        c = tl.get_clip_by_id(cid)
        mid = tl_start + c.duration_frames // 2
        tl.split_clip_at(cid, mid)
    t_split = time.perf_counter() - t0
    ok(f"split N/3 clips:    {fmt_ms(t_split)}   ({len(to_split)} splits)")

    # select + ripple delete every 5th
    t0 = time.perf_counter()
    victims = {tl.clips[i].id for i in range(0, tl.clip_count, 5)}
    tl.set_selection(victims)
    tl.ripple_delete_selected()
    t_ripple = time.perf_counter() - t0
    ok(f"ripple_delete:      {fmt_ms(t_ripple)}   ({len(victims)} victims)")

    # undo + redo
    t0 = time.perf_counter()
    tl.undo()
    t_undo = time.perf_counter() - t0
    t0 = time.perf_counter()
    tl.redo()
    t_redo = time.perf_counter() - t0
    ok(f"undo / redo:        {fmt_ms(t_undo)} / {fmt_ms(t_redo)}")

    # position math (hot path for paint — worth timing directly)
    ids = [c.id for c in tl.clips[::50]]
    t0 = time.perf_counter()
    for _ in range(10):
        for cid in ids:
            tl.get_clip_timeline_start(cid)
    t_posmath = (time.perf_counter() - t0) / (10 * len(ids))
    ok(f"get_clip_timeline_start: {fmt_ms(t_posmath)} per call "
       f"(O(n) over {tl.clip_count} items)")

    # invariants
    total = tl.get_total_duration_frames()
    assert total > 0, "total duration should be > 0"
    # empty add should no-op (Batch 1 fix)
    before_undo_len = len(tl._undo_stack)
    tl.add_clips([])
    assert len(tl._undo_stack) == before_undo_len, "empty add_clips pushed undo"
    ok(f"empty add_clips no-ops: stack len unchanged ({before_undo_len})")

    # in/out guard (Batch 1 fix): setting an invalid In shouldn't wipe Out
    tl.clear_in_out()
    tl.set_out_point(500)
    tl.set_in_point(600)  # invalid — In >= Out
    assert tl.out_point == 500, f"Out was wiped: {tl.out_point}"
    assert tl.in_point is None, f"In should not have been set: {tl.in_point}"
    ok("in/out guard: invalid In rejected, Out preserved")

    # project save/load round-trip
    td = Path(tempfile.mkdtemp(prefix="psynth_bench_"))
    try:
        pfile = td / "round.psynth"
        sources = {src_id: _FakeSource(src_id)}
        t0 = time.perf_counter()
        save_project(str(pfile), sources, tl.clips, playhead=0,
                     in_point=tl.in_point, out_point=tl.out_point)
        t_save = time.perf_counter() - t0
        t0 = time.perf_counter()
        data = load_project(str(pfile))
        t_load = time.perf_counter() - t0
        assert len(data["clips"]) == tl.clip_count
        size_kb = pfile.stat().st_size // 1024
        ok(f"project save / load ({tl.clip_count} clips, "
           f"{size_kb} KB): {fmt_ms(t_save)} / {fmt_ms(t_load)}")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("model", passed=True, timings={
        "add": t_add, "split": t_split, "ripple": t_ripple,
        "undo": t_undo, "redo": t_redo,
        "get_start_per_call": t_posmath,
    })


@dataclass
class _FakeSource:
    """Stand-in for VideoSource in model tests — only the id/fields used by save_project matter."""
    id: str
    file_path: str = "fake.mov"
    total_frames: int = 100000
    fps: float = 24.0
    width: int = 3840
    height: int = 2160
    codec: str = "hevc"


# -------------------------- section 4: probe ------------------------------

def section_probe(videos: List[Path]) -> SectionResult:
    header(f"Video probe ({len(videos)} file(s))")
    from utils.ffprobe import probe_video, probe_hdr

    warnings = 0
    for v in videos:
        subheader(v.name)
        if not v.exists():
            fail(f"file not found: {v}")
            warnings += 1
            continue
        info(f"size: {v.stat().st_size / (1024 ** 3):.2f} GB")
        t0 = time.perf_counter()
        vi = probe_video(str(v))
        t_probe = time.perf_counter() - t0
        if vi is None:
            fail(f"probe_video returned None")
            warnings += 1
            continue
        ok(f"{vi.width}x{vi.height} {vi.codec} @ {vi.fps:.3f} fps, "
           f"{vi.total_frames} frames, {vi.duration_seconds:.1f}s  "
           f"(probe: {fmt_ms(t_probe)})")
        is_hdr = probe_hdr(str(v))
        ok(f"HDR: {is_hdr}")

        # GOP structure — key for seek-latency prediction
        gop = _probe_gop(v, 30)
        if gop:
            ok(f"GOP (first 30s): {gop['i_frames']} I-frames, "
               f"avg={gop['avg_sec']:.2f}s (~{gop['avg_frames']:.0f} frames), "
               f"max={gop['max_sec']:.2f}s")
        else:
            warn("GOP probe failed (ffprobe timed out or no I-frames found)")
            warnings += 1
    return SectionResult("probe", passed=warnings == 0, warnings=warnings)


def _probe_gop(path: Path, duration_s: int) -> Optional[dict]:
    # 4K HEVC with DV metadata can take 30+ seconds to probe a 30s window.
    # Give it ample time.
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-read_intervals", f"%+{duration_s}",
           "-show_entries", "frame=pict_type,pts_time",
           "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    last = -1.0
    gaps = []
    ic = 0
    for line in out.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            pts = float(parts[0])
            ptype = parts[1]
        except ValueError:
            continue
        if ptype == "I":
            if last >= 0:
                gaps.append(pts - last)
            last = pts
            ic += 1
    if not gaps:
        return None
    avg = statistics.mean(gaps)
    return {"i_frames": ic, "avg_sec": avg,
            "avg_frames": avg * 24.0, "max_sec": max(gaps)}


# ---------------------- section 5: thumbnail decode -----------------------

def section_thumb(videos: List[Path]) -> SectionResult:
    header("Thumbnail decode benchmark")
    from utils.ffprobe import probe_video
    import av  # type: ignore
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    warnings = 0
    timings = {}

    for v in videos:
        subheader(v.name)
        vi = probe_video(str(v))
        if vi is None:
            warn("probe failed, skipping")
            warnings += 1
            continue
        total = vi.total_frames
        if total < 200:
            warn("too short, skipping")
            warnings += 1
            continue

        THUMB_W, THUMB_H = 192, 108
        single_frames = random.Random(0).sample(range(0, total), 10)
        sweep_base = total // 3
        sweep_targets = [sweep_base + i * 48 for i in range(8)]  # 8 frames within ~400-frame window

        # --- single-frame ffmpeg seek ---
        single_times = []
        for f in single_frames:
            ts = f / vi.fps
            cmd = ["ffmpeg", "-nostdin", "-v", "quiet",
                   "-ss", f"{ts:.4f}", "-i", str(v),
                   "-frames:v", "1",
                   "-vf", f"scale={THUMB_W}:{THUMB_H}:flags=fast_bilinear",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, timeout=30)
            dt = time.perf_counter() - t0
            if len(p.stdout) == THUMB_W * THUMB_H * 3:
                single_times.append(dt)
        if single_times:
            med = statistics.median(single_times)
            p90 = sorted(single_times)[int(len(single_times) * 0.9) - 1]
            ok(f"single-seek ffmpeg x{len(single_times)}: "
               f"median={fmt_ms(med)}  p90={fmt_ms(p90)}  "
               f"max={fmt_ms(max(single_times))}")
            timings[f"{v.name}:single"] = med
        else:
            fail("single-seek produced no output")
            warnings += 1

        # --- PyAV sweep decode ---
        try:
            t0 = time.perf_counter()
            container = av.open(str(v))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            target_pts = [int(f / vi.fps / tb) for f in sweep_targets]
            container.seek(target_pts[0], stream=stream)
            idx = 0
            frames_out = 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                while idx < len(target_pts) and frame.pts >= target_pts[idx]:
                    rgb = frame.to_ndarray(format="rgb24")
                    _ = cv2.resize(rgb, (THUMB_W, THUMB_H),
                                   interpolation=cv2.INTER_AREA)
                    frames_out += 1
                    idx += 1
                if idx >= len(target_pts):
                    break
            container.close()
            dt_sweep = time.perf_counter() - t0
            per = dt_sweep / max(frames_out, 1)
            ok(f"PyAV sweep ({frames_out}/{len(sweep_targets)} frames in one seek): "
               f"total={fmt_ms(dt_sweep)}  per-frame={fmt_ms(per)}")
            timings[f"{v.name}:sweep_per_frame"] = per
            if single_times:
                speedup = statistics.median(single_times) / per
                info(f"  sweep is {speedup:.1f}x faster per-frame than single-seek")
        except Exception as e:
            fail(f"PyAV sweep failed: {e}")
            warnings += 1

    return SectionResult("thumb", passed=warnings == 0,
                         warnings=warnings, timings=timings)


# --------------------- section 6: scene-detect decode ---------------------

def section_decode(videos: List[Path], duration_s: int) -> SectionResult:
    header(f"Scene-detection decode paths (first {duration_s}s)")
    from utils.ffprobe import probe_video

    warnings = 0
    timings = {}

    for v in videos:
        subheader(v.name)
        vi = probe_video(str(v))
        if vi is None:
            warn("probe failed, skipping")
            warnings += 1
            continue
        frames_needed = int(duration_s * vi.fps)

        # NVDEC + scale_cuda (fastest scene-detect path)
        nvdec_cuda = _time_decode(v, duration_s, frames_needed,
                                  hwaccel_cuda=True, scale_cuda=True)
        # NVDEC + CPU scale
        nvdec_cpu = _time_decode(v, duration_s, frames_needed,
                                 hwaccel_cuda=True, scale_cuda=False)
        # CPU only
        cpu_only = _time_decode(v, duration_s, frames_needed,
                                hwaccel_cuda=False, scale_cuda=False)

        for label, res in (("NVDEC+scale_cuda", nvdec_cuda),
                           ("NVDEC+CPU scale ", nvdec_cpu),
                           ("CPU only        ", cpu_only)):
            if res is None:
                warn(f"{label}: unavailable")
                warnings += 1
                continue
            fps = res["frames"] / res["sec"] if res["sec"] > 0 else 0
            ok(f"{label}: {res['frames']} frames in {fmt_ms(res['sec'])} "
               f"= {fps:6.0f} fps")
            timings[f"{v.name}:{label.strip()}"] = fps

    return SectionResult("decode", passed=warnings == 0,
                         warnings=warnings, timings=timings)


def _time_decode(path: Path, duration_s: int, frames_needed: int,
                 hwaccel_cuda: bool, scale_cuda: bool) -> Optional[dict]:
    """Emulate SceneDetector's 48x27 raw-frame decode path."""
    W, H = 48, 27
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if hwaccel_cuda:
        cmd += ["-hwaccel", "cuda"]
        if scale_cuda:
            cmd += ["-hwaccel_output_format", "cuda"]
    cmd += ["-ss", "0", "-t", str(duration_s), "-i", str(path)]
    if scale_cuda:
        cmd += ["-vf", f"scale_cuda={W}:{H}:format=yuv420p,hwdownload,"
                       f"format=yuv420p,format=rgb24"]
    else:
        cmd += ["-vf", f"scale={W}:{H}:flags=fast_bilinear,format=rgb24"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return None
    frame_size = W * H * 3
    frames = len(p.stdout) // frame_size
    if frames == 0:
        return None
    return {"frames": frames, "sec": dt}


# ------------------------ section 7: export -------------------------------

# Subset of VIDEO_PRESETS chosen for time + coverage. Skips heavy ProRes 4444 /
# FFV1 by default to keep runtime reasonable; pass --full-export to run all.
EXPORT_PRESETS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-cq", "23", "-pix_fmt", "yuv420p"],
    "h265_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-cq", "23", "-pix_fmt", "yuv420p"],
    "h264_cpu":   ["-c:v", "libx264", "-preset", "medium", "-crf", "23",
                   "-pix_fmt", "yuv420p"],
    "h265_cpu":   ["-c:v", "libx265", "-preset", "medium", "-crf", "23",
                   "-pix_fmt", "yuv420p"],
    "prores_lt":  ["-c:v", "prores_aw", "-profile:v", "1",
                   "-pix_fmt", "yuv422p10le"],
}

EXPORT_EXT = {
    "h264_nvenc": ".mp4", "h265_nvenc": ".mp4",
    "h264_cpu":   ".mp4", "h265_cpu":   ".mp4",
    "prores_lt":  ".mov",
}


def section_export(videos: List[Path], duration_s: int,
                   output_w: int = 1920, output_h: int = 1080) -> SectionResult:
    header(f"Export encode benchmark ({duration_s}s, {output_w}x{output_h})")
    from utils.ffprobe import probe_video, probe_hdr

    warnings = 0
    timings = {}
    td = Path(tempfile.mkdtemp(prefix="psynth_export_bench_"))
    try:
        for v in videos:
            subheader(v.name)
            vi = probe_video(str(v))
            if vi is None:
                warn("probe failed, skipping")
                warnings += 1
                continue
            is_hdr = probe_hdr(str(v))
            if is_hdr:
                info("source is HDR — output will tonemap via zscale (CPU)")

            frames = int(duration_s * vi.fps)

            for key, codec_args in EXPORT_PRESETS.items():
                is_nvenc = "nvenc" in codec_args[1]
                out_path = td / f"{v.stem}_{key}{EXPORT_EXT[key]}"
                cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
                if is_nvenc and not is_hdr:
                    # Zero-copy SDR NVENC path — the one Batch 3 wants to fix
                    cmd += ["-hwaccel", "cuda",
                            "-hwaccel_output_format", "cuda",
                            "-ss", "0", "-i", str(v),
                            "-vf", f"scale_cuda={output_w}:{output_h}:"
                                   f"format=yuv420p"]
                elif is_hdr:
                    # CPU tonemap path
                    cmd += ["-ss", "0", "-i", str(v),
                            "-vf",
                            f"zscale=t=linear:npl=100,format=gbrpf32le,"
                            f"zscale=p=bt709,tonemap=hable,"
                            f"zscale=t=bt709:m=bt709:r=tv,"
                            f"format=yuv420p,scale={output_w}:{output_h}"]
                else:
                    cmd += ["-ss", "0", "-i", str(v),
                            "-vf", f"scale={output_w}:{output_h}"]
                cmd += ["-frames:v", str(frames), "-r", f"{vi.fps:.6f}"]
                cmd += codec_args
                if is_hdr:
                    cmd += ["-colorspace", "bt709", "-color_trc", "bt709",
                            "-color_primaries", "bt709"]
                cmd += ["-an", str(out_path)]

                t0 = time.perf_counter()
                try:
                    p = subprocess.run(cmd, capture_output=True,
                                       text=True, timeout=600)
                except subprocess.TimeoutExpired:
                    warn(f"{key}: timed out after 600s")
                    warnings += 1
                    continue
                dt = time.perf_counter() - t0
                if p.returncode != 0:
                    warn(f"{key}: ffmpeg failed — "
                         f"{(p.stderr or '').splitlines()[-1:] or ['?']}")
                    warnings += 1
                    continue
                if not out_path.exists() or out_path.stat().st_size == 0:
                    warn(f"{key}: empty output")
                    warnings += 1
                    continue
                size_mb = out_path.stat().st_size / (1024 * 1024)
                realtime = duration_s / dt if dt > 0 else 0
                ok(f"{key:11s}: {fmt_ms(dt)}  "
                   f"({realtime:5.1f}x realtime)  {size_mb:6.1f} MB")
                timings[f"{v.name}:{key}"] = dt
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export", passed=warnings == 0,
                         warnings=warnings, timings=timings)


# ------------------------------- main -------------------------------------

def main():
    _enable_win_ansi()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", default=[],
                        help="Path to a sample video (repeatable)")
    parser.add_argument("--duration", type=int, default=5,
                        help="Seconds of video to decode/encode in benchmarks (default 5)")
    parser.add_argument("--skip", action="append", default=[],
                        choices=SECTIONS, help="Skip a section (repeatable)")
    args = parser.parse_args()

    videos = [Path(v) for v in args.video]
    for v in videos:
        if not v.exists():
            print(f"{Colors.R}Video not found: {v}{Colors.X}")
            sys.exit(2)

    started = time.perf_counter()

    if "env" not in args.skip:
        run_section("env", section_env)
    if "compile" not in args.skip:
        run_section("compile", section_compile)
    if "model" not in args.skip:
        run_section("model", section_model)

    if videos:
        if "probe" not in args.skip:
            run_section("probe", lambda: section_probe(videos))
        if "thumb" not in args.skip:
            run_section("thumb", lambda: section_thumb(videos))
        if "decode" not in args.skip:
            run_section("decode", lambda: section_decode(videos, args.duration))
        if "export" not in args.skip:
            run_section("export",
                        lambda: section_export(videos, args.duration))
    else:
        print(f"\n{Colors.Y}No --video provided — skipping probe/thumb/"
              f"decode/export sections.{Colors.X}")

    # ---- summary ----
    elapsed = time.perf_counter() - started
    header("Summary")
    any_fail = False
    for r in RESULTS:
        mark = (f"{Colors.G}PASS{Colors.X}" if r.passed
                else f"{Colors.R}FAIL{Colors.X}")
        warn_tag = f" ({r.warnings} warn)" if r.warnings else ""
        note = f" — {r.note}" if r.note else ""
        print(f"  [{mark}] {r.name:<8}{warn_tag}{note}")
        if not r.passed:
            any_fail = True
    print(f"\nTotal time: {elapsed:.1f}s")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
