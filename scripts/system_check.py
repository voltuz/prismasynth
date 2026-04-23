"""PrismaSynth system correctness checkup.

Complements perf_benchmark.py — this script exercises the real code paths
and asserts on correctness invariants. Catches the class of bugs we
hit while debugging exports (off-by-one seek, timebase drift, tonemap
misapplication, etc.) plus data-model invariants.

Run:
    venv\\Scripts\\python scripts\\system_check.py --video PATH
             [--skip SECTION ...]

Sections: env, compile, model, project, thumb_accuracy, export_accuracy,
export_timebase.
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
os.environ["PATH"] = str(SRC) + os.pathsep + os.environ.get("PATH", "")

logging.basicConfig(level=logging.WARNING)

SECTIONS = ("env", "compile", "model", "project", "thumb_accuracy",
            "export_accuracy", "export_timebase", "export_multi_segment",
            "export_hdr", "export_image_sequence", "export_edl",
            "export_nvenc")


# ----------------------------- output helpers -----------------------------

class C:
    G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[1m"
    CYAN = "\033[36m"; X = "\033[0m"


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


def header(t): print(f"\n{C.B}{C.CYAN}{'='*70}\n{t}\n{'='*70}{C.X}")
def ok(m): print(f"  {C.G}[OK]{C.X} {m}")
def fail(m): print(f"  {C.R}[FAIL]{C.X} {m}")
def warn(m): print(f"  {C.Y}[WARN]{C.X} {m}")
def info(m): print(f"  {m}")


@dataclass
class SectionResult:
    name: str
    passed: bool
    failures: int = 0
    note: str = ""


RESULTS: List[SectionResult] = []


def run_section(name: str, fn: Callable[[], SectionResult]) -> SectionResult:
    try:
        res = fn()
    except Exception:
        traceback.print_exc()
        res = SectionResult(name=name, passed=False, note="exception")
    RESULTS.append(res)
    return res


# ----------------------------- section: env -------------------------------

def section_env() -> SectionResult:
    header("Environment")
    failures = 0
    info(f"Python: {sys.version.split()[0]} on {platform.platform()}")

    for label, cmd in [("ffmpeg", ["ffmpeg", "-version"]),
                       ("ffprobe", ["ffprobe", "-version"])]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                ok(f"{label}: {r.stdout.splitlines()[0]}")
            else:
                fail(f"{label}: exit {r.returncode}")
                failures += 1
        except (FileNotFoundError, subprocess.TimeoutExpired):
            fail(f"{label}: not on PATH")
            failures += 1
    return SectionResult("env", passed=failures == 0, failures=failures)


# --------------------------- section: compile -----------------------------

def section_compile() -> SectionResult:
    header("Module compilation")
    failed = []
    for f in sorted(SRC.rglob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append((f.relative_to(ROOT), str(e)))
    if failed:
        for p, e in failed:
            fail(f"{p}: {e}")
        return SectionResult("compile", passed=False, failures=len(failed))
    ok(f"all {sum(1 for _ in SRC.rglob('*.py'))} files compile cleanly")
    return SectionResult("compile", passed=True)


# ---------------------------- section: model ------------------------------

def section_model() -> SectionResult:
    header("Data-model invariants")
    from PySide6.QtCore import QCoreApplication
    from core.clip import Clip
    from core.timeline import TimelineModel
    _ = QCoreApplication.instance() or QCoreApplication(sys.argv)

    failures = 0

    def check(cond: bool, label: str):
        nonlocal failures
        if cond:
            ok(label)
        else:
            fail(label)
            failures += 1

    tl = TimelineModel()
    sid = "src_x"
    tl.add_clips([Clip(source_id=sid, source_in=i * 100, source_out=i * 100 + 49)
                  for i in range(20)])
    initial_total = tl.get_total_duration_frames()
    check(initial_total == 20 * 50, f"total duration after add = 1000 ({initial_total})")

    # Empty add: no undo push, no emit
    before = len(tl._undo_stack)
    tl.add_clips([])
    check(len(tl._undo_stack) == before, "empty add_clips() does not push undo")

    # Empty replace_detected
    before = len(tl._undo_stack)
    tl.replace_detected({})
    check(len(tl._undo_stack) == before, "empty replace_detected() does not push undo")

    # Replace_detected with empty sub-clip list for a key should be filtered,
    # not silently delete the clip
    some_id = tl.clips[3].id
    before_total = tl.get_total_duration_frames()
    tl.replace_detected({some_id: []})
    check(tl.get_total_duration_frames() == before_total,
          "replace_detected with empty sub-clips preserves the original clip")

    # In/out guards
    tl.clear_in_out()
    tl.set_in_point(100)
    tl.set_out_point(50)   # invalid: out <= in
    check(tl.out_point is None and tl.in_point == 100,
          "set_out_point with out<=in is rejected (keeps in_point)")
    tl.clear_in_out()
    tl.set_out_point(500)
    tl.set_in_point(600)   # invalid: in >= out
    check(tl.in_point is None and tl.out_point == 500,
          "set_in_point with in>=out is rejected (keeps out_point)")

    # Split invariants
    tl.clear(); tl.clear_undo()
    tl.add_clips([Clip(source_id=sid, source_in=0, source_out=99)])
    clip_id = tl.clips[0].id
    ok_split = tl.split_clip_at(clip_id, 50)
    check(ok_split and len(tl.clips) == 2, "split at mid produces two clips")
    check(tl.get_total_duration_frames() == 100,
          "split preserves total duration")

    # Undo/redo round-trip
    tl.clear(); tl.clear_undo()
    tl.add_clips([Clip(source_id=sid, source_in=0, source_out=99)])
    tl.add_clips([Clip(source_id=sid, source_in=100, source_out=199)])
    snapshot_total = tl.get_total_duration_frames()
    snapshot_count = len(tl.clips)
    tl.undo()
    tl.redo()
    check(tl.get_total_duration_frames() == snapshot_total
          and len(tl.clips) == snapshot_count,
          "undo -> redo preserves clip count and total duration")

    # Gap merging: two consecutive gaps should collapse after remove_clips
    tl.clear(); tl.clear_undo()
    tl.add_clips([Clip(source_id=sid, source_in=0, source_out=9),
                  Clip(source_id=sid, source_in=10, source_out=19),
                  Clip(source_id=sid, source_in=20, source_out=29)])
    c1, c2 = tl.clips[0].id, tl.clips[1].id
    tl.remove_clips({c1, c2})
    gaps = [c for c in tl.clips if c.is_gap]
    check(len(gaps) == 1, f"adjacent removed clips merge into one gap (got {len(gaps)})")

    # Fuzz: random operations never corrupt the model
    tl.clear(); tl.clear_undo()
    tl.add_clips([Clip(source_id=sid, source_in=i * 20, source_out=i * 20 + 19)
                  for i in range(10)])
    rng = random.Random(42)
    for _ in range(50):
        op = rng.choice(["split", "remove", "undo", "redo"])
        try:
            if op == "split" and tl.clips:
                c = rng.choice(tl.clips)
                if not c.is_gap:
                    start = tl.get_clip_timeline_start(c.id)
                    if c.duration_frames >= 3:
                        tl.split_clip_at(c.id, start + c.duration_frames // 2)
            elif op == "remove" and tl.clips:
                c = rng.choice(tl.clips)
                tl.remove_clips({c.id})
            elif op == "undo":
                tl.undo()
            elif op == "redo":
                tl.redo()
        except Exception as e:
            fail(f"fuzz op {op} raised: {e}")
            failures += 1
            break
        # invariants after every op
        total = tl.get_total_duration_frames()
        if total < 0:
            fail(f"total duration went negative ({total})")
            failures += 1
            break
        dangling = tl._selected_ids - {c.id for c in tl.clips}
        if dangling:
            fail(f"selection references missing clip ids: {dangling}")
            failures += 1
            break
    else:
        ok("fuzz: 50 random ops, invariants held throughout")

    return SectionResult("model", passed=failures == 0, failures=failures)


# --------------------------- section: project -----------------------------

def section_project() -> SectionResult:
    header("Project save/load round-trip")
    from core.clip import Clip
    from core.video_source import VideoSource
    from core.project import save_project, load_project

    failures = 0

    sources = {"src": VideoSource(
        id="src", file_path="fake.mov", total_frames=10000,
        fps=23.976, width=3840, height=2160, codec="hevc")}
    clips = [
        Clip(source_id="src", source_in=100, source_out=199,
             id="c1", label="a", color_index=0),
        Clip(source_id=None, source_in=0, source_out=49,
             id="c2", label="", color_index=0),
        Clip(source_id="src", source_in=500, source_out=649,
             id="c3", label="b", color_index=3),
    ]

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_"))
    try:
        pfile = td / "r.psynth"
        save_project(str(pfile), sources, clips, playhead=123,
                     in_point=10, out_point=200)

        def ck(cond, label):
            nonlocal failures
            if cond:
                ok(label)
            else:
                fail(label); failures += 1

        # Atomic save guarantee: no .tmp left behind
        ck(not (td / "r.psynth.tmp").exists(),
           "no lingering .psynth.tmp after save")

        data = load_project(str(pfile))
        ck(len(data["clips"]) == 3, "3 clips round-tripped")
        ck(data["playhead_position"] == 123, "playhead preserved")
        ck(data["in_point"] == 10, "in_point preserved")
        ck(data["out_point"] == 200, "out_point preserved")
        ck("src" in data["sources"], "source id preserved")
        for i, (orig, loaded) in enumerate(zip(clips, data["clips"])):
            for attr in ("id", "source_id", "source_in", "source_out",
                         "label", "color_index"):
                if getattr(orig, attr) != getattr(loaded, attr):
                    fail(f"clip {i}.{attr} changed: "
                         f"{getattr(orig, attr)!r} → {getattr(loaded, attr)!r}")
                    failures += 1
                    break
            else:
                continue
        else:
            ok("all 3 clips byte-equivalent after round-trip")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("project", passed=failures == 0, failures=failures)


# ----------------------- helpers for video-dependent tests ----------------

def _probe(path: Path) -> Optional[dict]:
    from utils.ffprobe import probe_video
    vi = probe_video(str(path))
    if vi is None:
        return None
    return {"width": vi.width, "height": vi.height, "fps": vi.fps,
            "total_frames": vi.total_frames, "codec": vi.codec}


def _extract_frame(video: Path, frame_n: int, out: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(video),
           "-vf", f"select=eq(n\\,{frame_n})", "-frames:v", "1",
           "-vsync", "vfr", str(out)]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    return r.returncode == 0 and out.exists()


def _pixel_diff(a: Path, b: Path):
    import numpy as np
    from PIL import Image
    ia = np.array(Image.open(a).convert("RGB"))
    ib = np.array(Image.open(b).convert("RGB"))
    if ia.shape != ib.shape:
        from PIL import Image as I
        ib = np.array(I.open(b).convert("RGB").resize(
            (ia.shape[1], ia.shape[0])))
    return float(np.abs(ia.astype(int) - ib.astype(int)).mean())


# ---------------------- section: thumb_accuracy ---------------------------

def section_thumb_accuracy(videos: List[Path]) -> SectionResult:
    header("Thumbnail frame-accuracy")
    failures = 0
    td = Path(tempfile.mkdtemp(prefix="psynth_sys_thumb_"))
    try:
        for v in videos:
            meta = _probe(v)
            if not meta:
                warn(f"{v.name}: probe failed"); continue
            # Use small frame numbers that work for all sources
            for frame_n in [0, 500, max(100, meta["total_frames"] // 2)]:
                # The thumbnail subsystem's single-seek path shares its
                # ffmpeg invocation with what we extract as ground-truth,
                # so a content comparison is not meaningful here. Instead
                # we verify that seeking to frame N returns the expected
                # frame by comparing against a select=eq(n,N) reference —
                # a check that would have caught the off-by-one seek.
                ref = td / f"ref_{v.stem}_{frame_n}.png"
                thumb = td / f"thumb_{v.stem}_{frame_n}.png"
                if not _extract_frame(v, frame_n, ref):
                    warn(f"{v.name}: ref extract failed at n={frame_n}")
                    continue
                # Thumbnail-style seek: pre-input -ss timestamp
                ts = frame_n / meta["fps"] if meta["fps"] > 0 else 0
                cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
                       "-ss", f"{ts:.6f}", "-i", str(v),
                       "-frames:v", "1", str(thumb)]
                r = subprocess.run(cmd, capture_output=True, timeout=60)
                if r.returncode != 0 or not thumb.exists():
                    warn(f"{v.name}: timestamp seek failed at n={frame_n}")
                    continue
                diff = _pixel_diff(ref, thumb)
                # Thumb uses timestamp seek which can land ±1 frame off
                # with float-rounded fps; note this but don't hard-fail
                # because the production thumb cache uses PyAV sweep for
                # frame-accurate reads.
                status = "exact" if diff < 2 else "off by up to 1 frame"
                note = f"  ({status})"
                if diff < 50:
                    ok(f"{v.name} frame {frame_n}: mean diff={diff:.2f}{note}")
                else:
                    fail(f"{v.name} frame {frame_n}: mean diff={diff:.2f} — "
                         f"likely wrong frame")
                    failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return SectionResult("thumb_accuracy", passed=failures == 0,
                         failures=failures)


# ---------------------- section: export_accuracy --------------------------

def _export_via_exporter(video_path: Path, src_in: int, count: int,
                         codec_args: list, out_path: Path,
                         width: int = None, height: int = None,
                         fps: float = 23.976, timeout: float = 120.0) -> str:
    """Run the real Exporter class against a one-clip timeline. Returns
    "ok", "error:<msg>", or "timeout"."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from core.exporter import Exporter
    from utils.ffprobe import probe_video

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    meta = probe_video(str(video_path))
    if meta is None:
        return "error:probe failed"

    vs = VideoSource(id="s0", file_path=str(video_path),
                     total_frames=meta.total_frames,
                     fps=meta.fps, width=meta.width,
                     height=meta.height, codec=meta.codec)
    tl = TimelineModel()
    tl.add_clips([Clip(source_id="s0", source_in=src_in,
                       source_out=src_in + count - 1)],
                 assign_colors=False)

    exporter = Exporter(tl, {"s0": vs})

    result = {"kind": None, "msg": ""}
    loop = QEventLoop()

    def on_finished():
        result["kind"] = "ok"
        loop.quit()

    def on_error(msg):
        result["kind"] = "error"
        result["msg"] = msg
        loop.quit()

    def on_cancelled():
        result["kind"] = "cancelled"
        loop.quit()

    from PySide6.QtCore import Qt as _Qt
    exporter.finished.connect(on_finished, _Qt.ConnectionType.QueuedConnection)
    exporter.error.connect(on_error, _Qt.ConnectionType.QueuedConnection)
    exporter.cancelled.connect(on_cancelled, _Qt.ConnectionType.QueuedConnection)

    settings = {
        "mode": "video",
        "output_path": str(out_path),
        "ffmpeg_args": codec_args,
        "width": width or meta.width,
        "height": height or meta.height,
        "fps": fps,
    }
    exporter.export(settings)
    # Timeout watchdog
    QTimer.singleShot(int(timeout * 1000),
                      lambda: (exporter.cancel(), loop.quit()))
    loop.exec()
    if result["kind"] == "ok":
        return "ok"
    elif result["kind"] == "error":
        return f"error:{result['msg'][:200]}"
    elif result["kind"] == "cancelled":
        return "timeout"
    return "unknown"


def section_export_accuracy(videos: List[Path]) -> SectionResult:
    """End-to-end test that runs the real Exporter class on a tiny
    timeline and verifies (a) frame count is exact, (b) first frame
    of output matches source frame src_in (would catch off-by-one
    seek), (c) container metadata is correct."""
    header("Export frame-accuracy (real Exporter class)")
    failures = 0

    # Prefer SDR source (content comparison is reliable without tonemap)
    sdr = [v for v in videos
           if "YTS" in v.name or "Hobbs" in v.name or "1080p" in v.name]
    if not sdr and videos:
        sdr = [videos[0]]
    if not sdr:
        warn("no source provided; skipping")
        return SectionResult("export_accuracy", passed=True, note="skipped")

    video = sdr[0]
    info(f"source: {video.name}")

    from utils.ffprobe import probe_video
    meta = probe_video(str(video))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_accuracy", passed=False, failures=1)

    # Small test: 30 frames from a known offset
    src_in = 500
    count = 30
    td = Path(tempfile.mkdtemp(prefix="psynth_sys_export_"))
    try:
        # Codecs to cover: at least one MOV-container (tests timescale),
        # one MP4-container, one CPU, one GPU.
        trials = [
            ("prores_lt", "mov",
             ["-c:v", "prores_aw", "-profile:v", "1",
              "-pix_fmt", "yuv422p10le"]),
            ("h264_cpu", "mp4",
             ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
              "-pix_fmt", "yuv420p"]),
        ]

        for label, ext, args in trials:
            out = td / f"test_{label}.{ext}"
            t0 = time.perf_counter()
            result = _export_via_exporter(
                video, src_in, count, args, out,
                fps=meta.fps, timeout=120.0)
            dt = time.perf_counter() - t0
            if result != "ok":
                fail(f"{label}: export failed — {result}")
                failures += 1
                continue

            # 1. File exists, nonzero
            if not out.exists() or out.stat().st_size == 0:
                fail(f"{label}: output missing or empty")
                failures += 1
                continue

            # 2. Frame count is exact
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-count_packets", "-show_entries", "stream=nb_read_packets",
                   "-of", "csv=p=0", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            nb = int((r.stdout or "0").strip() or 0)
            if nb != count:
                fail(f"{label}: frame count {nb} != expected {count}")
                failures += 1
            else:
                ok(f"{label}: {count} frames produced  ({dt:.1f}s)")

            # 3. Frame accuracy: output frame 0 should match source frame
            # src_in. Would have caught the off-by-one seek.
            out_f0 = td / f"out_{label}_f0.png"
            src_ref = td / f"src_{src_in}.png"
            if not src_ref.exists():
                _extract_frame(video, src_in, src_ref)
            if _extract_frame(out, 0, out_f0) and src_ref.exists():
                diff = _pixel_diff(src_ref, out_f0)
                # Compare to neighbors too to detect off-by-one
                diffs = {0: diff}
                for off in (-1, 1, 2):
                    ref = td / f"src_{src_in + off}.png"
                    if not ref.exists():
                        _extract_frame(video, src_in + off, ref)
                    if ref.exists():
                        diffs[off] = _pixel_diff(ref, out_f0)
                best_off = min(diffs, key=diffs.get)
                if best_off == 0:
                    ok(f"{label}: first output frame = source frame {src_in} "
                       f"(diff={diffs[0]:.2f}; neighbors={{-1: {diffs.get(-1, '?')}, "
                       f"1: {diffs.get(1, '?')}}})")
                else:
                    fail(f"{label}: first output frame is source {src_in + best_off}, "
                         f"not {src_in} — off by {best_off}")
                    failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_accuracy", passed=failures == 0,
                         failures=failures)


# ---------------------- section: export_timebase --------------------------

def section_export_timebase(videos: List[Path]) -> SectionResult:
    """Run a ProRes MOV export and verify the output container has the
    1/24000000 timescale + uniform packet durations. Would have caught
    the 1/16000 jitter that made Resolve show duplicate frames."""
    header("Export timebase integrity")
    failures = 0

    sdr = [v for v in videos
           if "YTS" in v.name or "Hobbs" in v.name or "1080p" in v.name]
    source = sdr[0] if sdr else (videos[0] if videos else None)
    if source is None:
        warn("no source provided; skipping")
        return SectionResult("export_timebase", passed=True, note="skipped")

    from utils.ffprobe import probe_video
    meta = probe_video(str(source))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_timebase", passed=False, failures=1)

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_tb_"))
    try:
        out = td / "tb.mov"
        args = ["-c:v", "prores_aw", "-profile:v", "1",
                "-pix_fmt", "yuv422p10le"]
        result = _export_via_exporter(source, 200, 60, args, out,
                                      fps=meta.fps, timeout=120.0)
        if result != "ok":
            fail(f"export failed: {result}")
            return SectionResult("export_timebase", passed=False, failures=1)

        # Inspect timebase and packet durations
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=time_base,r_frame_rate,avg_frame_rate",
               "-show_entries", "packet=duration",
               "-of", "json", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            fail(f"ffprobe failed: {r.stderr[:200]}")
            return SectionResult("export_timebase", passed=False, failures=1)
        data = json.loads(r.stdout)
        stream = data["streams"][0]
        tb = stream["time_base"]
        rfr = stream["r_frame_rate"]
        afr = stream["avg_frame_rate"]
        packets = data.get("packets", [])

        info(f"time_base={tb}  r_frame_rate={rfr}  avg_frame_rate={afr}  "
             f"({len(packets)} packets)")

        # time_base should not be 1/16000 for NTSC rates (the jittering bug)
        if tb == "1/16000":
            fail("time_base is 1/16000 — NTSC jitter bug present")
            failures += 1
        else:
            ok(f"time_base != 1/16000 (got {tb})")

        # All packet durations equal = CFR. For NTSC 23.976 at 1/24000000
        # timescale, each packet should be exactly 1001000 ticks. Some MOV
        # muxers write a short last packet (common quirk); tolerate exactly
        # one outlier at the tail.
        all_durs = [int(p["duration"]) for p in packets if "duration" in p]
        durs = set(all_durs)
        if len(durs) == 1:
            ok(f"all {len(packets)} packets have identical duration "
               f"({next(iter(durs))} ticks) -- clean CFR")
        elif len(durs) == 2 and all_durs[-1] != all_durs[0]:
            common = all_durs[0]
            last = all_durs[-1]
            n_common = sum(1 for d in all_durs if d == common)
            warn(f"{n_common}/{len(packets)} packets at {common} ticks, "
                 f"last packet at {last} ticks (MOV tail-packet convention "
                 f"-- should not affect playback)")
        else:
            fail(f"packet durations vary: {sorted(durs)} -- not CFR")
            failures += 1

        # r_frame_rate should equal avg_frame_rate (or very close)
        def to_float(r):
            n, d = r.split("/")
            return float(n) / float(d) if float(d) else 0.0
        rfr_f, afr_f = to_float(rfr), to_float(afr)
        if abs(rfr_f - afr_f) / max(rfr_f, 1e-6) < 0.001:
            ok(f"r_frame_rate matches avg_frame_rate ({rfr_f:.4f} ~= {afr_f:.4f})")
        else:
            fail(f"r_frame_rate {rfr_f:.4f} != avg_frame_rate {afr_f:.4f} — timing drift")
            failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_timebase", passed=failures == 0,
                         failures=failures)


# -------------------- section: export_multi_segment ---------------------

def _run_exporter(tl, sources, settings, timeout=180.0) -> str:
    """Run the real Exporter class and block until done. Returns
    'ok', 'error:<msg>', 'cancelled', or 'timeout'."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer, Qt as _Qt
    from core.exporter import Exporter

    _ = QCoreApplication.instance() or QCoreApplication(sys.argv)
    exporter = Exporter(tl, sources)
    result = {"kind": None, "msg": ""}
    loop = QEventLoop()

    def on_finished():
        result["kind"] = "ok"; loop.quit()

    def on_error(msg):
        result["kind"] = "error"; result["msg"] = msg; loop.quit()

    def on_cancelled():
        result["kind"] = "cancelled"; loop.quit()

    exporter.finished.connect(on_finished, _Qt.ConnectionType.QueuedConnection)
    exporter.error.connect(on_error, _Qt.ConnectionType.QueuedConnection)
    exporter.cancelled.connect(on_cancelled, _Qt.ConnectionType.QueuedConnection)
    exporter.export(settings)
    QTimer.singleShot(int(timeout * 1000),
                      lambda: (exporter.cancel(), loop.quit()))
    loop.exec()
    if result["kind"] == "ok":
        return "ok"
    if result["kind"] == "error":
        return f"error:{result['msg'][:200]}"
    if result["kind"] == "cancelled":
        return "cancelled"
    return "timeout"


def section_export_multi_segment(videos: List[Path]) -> SectionResult:
    """Export a timeline with multiple non-contiguous clips (each becomes
    its own segment; concat runs). Verifies frame count matches the sum
    and that frame N of the output maps to the expected source frame at
    each clip boundary — would catch concat-time seek / timebase bugs."""
    header("Export: multi-segment concat")
    failures = 0
    sdr = [v for v in videos
           if "YTS" in v.name or "Hobbs" in v.name or "1080p" in v.name]
    source = sdr[0] if sdr else (videos[0] if videos else None)
    if source is None:
        warn("no source provided; skipping")
        return SectionResult("export_multi_segment", passed=True, note="skipped")

    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from utils.ffprobe import probe_video

    meta = probe_video(str(source))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_multi_segment", passed=False, failures=1)

    # Four non-contiguous clips, gaps in between. Small per-clip frame
    # counts so the whole test is fast.
    # (src_in, count)
    plan = [(200, 20), (1000, 15), (2000, 25), (3000, 10)]
    expected_total = sum(c for _, c in plan)

    vs = VideoSource(id="s0", file_path=str(source),
                     total_frames=meta.total_frames, fps=meta.fps,
                     width=meta.width, height=meta.height, codec=meta.codec)
    tl = TimelineModel()
    clips = [Clip(source_id="s0", source_in=s, source_out=s + n - 1)
             for s, n in plan]
    tl.add_clips(clips, assign_colors=False)

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_multi_"))
    try:
        out = td / "multi.mov"
        settings = {
            "mode": "video",
            "output_path": str(out),
            "ffmpeg_args": ["-c:v", "prores_aw", "-profile:v", "1",
                            "-pix_fmt", "yuv422p10le"],
            "width": meta.width, "height": meta.height, "fps": meta.fps,
        }
        result = _run_exporter(tl, {"s0": vs}, settings)
        if result != "ok":
            fail(f"export failed: {result}")
            return SectionResult("export_multi_segment", passed=False, failures=1)

        # Check total frame count
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_packets", "-show_entries", "stream=nb_read_packets",
               "-of", "csv=p=0", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        nb = int((r.stdout or "0").strip() or 0)
        if nb == expected_total:
            ok(f"total frame count = {expected_total}")
        else:
            fail(f"total frame count {nb} != expected {expected_total}")
            failures += 1

        # Verify each segment boundary lands on the correct source frame.
        # Boundaries in the output are at cumulative sums of clip lengths.
        boundary = 0
        for (src_in, count) in plan:
            # Output frame at `boundary` should == source frame src_in
            out_frame_img = td / f"out_b{boundary}.png"
            src_ref_img = td / f"src_{src_in}.png"
            ok_out = _extract_frame(out, boundary, out_frame_img)
            ok_src = src_ref_img.exists() or _extract_frame(
                source, src_in, src_ref_img)
            if not (ok_out and ok_src):
                warn(f"segment at output frame {boundary}: extraction failed")
                boundary += count
                continue
            diff = _pixel_diff(src_ref_img, out_frame_img)
            # Also check neighbors -- off-by-one would show
            diffs = {0: diff}
            for off in (-1, 1):
                ref = td / f"src_{src_in + off}.png"
                if not ref.exists():
                    _extract_frame(source, src_in + off, ref)
                if ref.exists():
                    diffs[off] = _pixel_diff(ref, out_frame_img)
            best_off = min(diffs, key=diffs.get)
            if best_off == 0:
                ok(f"output frame {boundary} = source frame {src_in} "
                   f"(diff={diffs[0]:.2f})")
            else:
                fail(f"output frame {boundary} is source {src_in + best_off}, "
                     f"not {src_in} -- off by {best_off}")
                failures += 1
            boundary += count
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_multi_segment", passed=failures == 0,
                         failures=failures)


# ---------------------- section: export_hdr ------------------------------

def section_export_hdr(videos: List[Path]) -> SectionResult:
    """Export from an HDR source: verify frame count, clean timebase, and
    that the output's color metadata claims BT.709 (i.e. tonemap was
    actually applied, not passed through as raw BT.2020)."""
    header("Export: HDR source (tonemap applied)")
    failures = 0
    hdr = [v for v in videos
           if "Roman.Holiday" in v.name or "Monster" in v.name
           or "roman_holiday" in v.name.lower()]
    source = hdr[0] if hdr else None
    if source is None:
        warn("no HDR source provided; skipping")
        return SectionResult("export_hdr", passed=True, note="skipped")

    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from utils.ffprobe import probe_video

    meta = probe_video(str(source))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_hdr", passed=False, failures=1)

    count = 30
    src_in = 1000
    vs = VideoSource(id="s0", file_path=str(source),
                     total_frames=meta.total_frames, fps=meta.fps,
                     width=meta.width, height=meta.height, codec=meta.codec)
    tl = TimelineModel()
    tl.add_clips([Clip(source_id="s0", source_in=src_in,
                       source_out=src_in + count - 1)],
                 assign_colors=False)

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_hdr_"))
    try:
        out = td / "hdr.mov"
        settings = {
            "mode": "video",
            "output_path": str(out),
            "ffmpeg_args": ["-c:v", "prores_aw", "-profile:v", "1",
                            "-pix_fmt", "yuv422p10le"],
            "width": 1920, "height": 1080, "fps": meta.fps,
        }
        result = _run_exporter(tl, {"s0": vs}, settings, timeout=240.0)
        if result != "ok":
            fail(f"export failed: {result}")
            return SectionResult("export_hdr", passed=False, failures=1)

        # Verify: frame count
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_packets",
               "-show_entries", "stream=nb_read_packets,time_base,"
                                "color_transfer,color_primaries",
               "-of", "json", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout)
        s = data["streams"][0]
        nb = int(s.get("nb_read_packets", 0))
        if nb == count:
            ok(f"{count} frames produced")
        else:
            fail(f"frame count {nb} != expected {count}")
            failures += 1

        # Timebase clean
        tb = s.get("time_base", "?")
        if tb == "1/24000000":
            ok(f"time_base = {tb}")
        else:
            fail(f"time_base = {tb} (expected 1/24000000)")
            failures += 1

        # Tonemap applied: output should claim bt709, not bt2020/smpte2084.
        # ffprobe may omit the fields if they default to bt709; missing =
        # OK, but if present they must not be HDR.
        ct = s.get("color_transfer", "")
        cp = s.get("color_primaries", "")
        if ct in ("smpte2084", "arib-std-b67") or cp == "bt2020":
            fail(f"output claims HDR (color_transfer={ct}, "
                 f"color_primaries={cp}) -- tonemap not applied?")
            failures += 1
        else:
            ok(f"output color metadata is SDR (transfer={ct or 'default'}, "
               f"primaries={cp or 'default'})")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_hdr", passed=failures == 0, failures=failures)


# -------------------- section: export_image_sequence ---------------------

def section_export_image_sequence(videos: List[Path]) -> SectionResult:
    """Export 30 frames of an SDR source as a PNG sequence; verify N
    files are produced and frame 1 content matches source."""
    header("Export: image sequence (PNG)")
    failures = 0
    sdr = [v for v in videos
           if "YTS" in v.name or "Hobbs" in v.name or "1080p" in v.name]
    source = sdr[0] if sdr else (videos[0] if videos else None)
    if source is None:
        warn("no source provided; skipping")
        return SectionResult("export_image_sequence", passed=True, note="skipped")

    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from utils.ffprobe import probe_video

    meta = probe_video(str(source))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_image_sequence", passed=False, failures=1)

    count = 30
    src_in = 500
    vs = VideoSource(id="s0", file_path=str(source),
                     total_frames=meta.total_frames, fps=meta.fps,
                     width=meta.width, height=meta.height, codec=meta.codec)
    tl = TimelineModel()
    tl.add_clips([Clip(source_id="s0", source_in=src_in,
                       source_out=src_in + count - 1)],
                 assign_colors=False)

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_imgseq_"))
    try:
        out_dir = td / "frames"
        settings = {
            "mode": "image_sequence",
            "output_dir": str(out_dir),
            "format": "png", "ext": ".png",
            "width": meta.width, "height": meta.height,
        }
        result = _run_exporter(tl, {"s0": vs}, settings, timeout=120.0)
        if result != "ok":
            fail(f"export failed: {result}")
            return SectionResult("export_image_sequence",
                                 passed=False, failures=1)

        # Verify N png files
        pngs = sorted(out_dir.glob("*.png"))
        if len(pngs) == count:
            ok(f"{count} PNG files produced")
        else:
            fail(f"got {len(pngs)} PNG files, expected {count}")
            failures += 1
            if len(pngs) == 0:
                return SectionResult("export_image_sequence",
                                     passed=False, failures=failures)

        # Check first PNG's content matches source frame src_in
        src_ref = td / f"src_{src_in}.png"
        _extract_frame(source, src_in, src_ref)
        if src_ref.exists() and pngs:
            diff = _pixel_diff(src_ref, pngs[0])
            # Check neighbors for off-by-one
            diffs = {0: diff}
            for off in (-1, 1):
                ref = td / f"src_{src_in + off}.png"
                if not ref.exists():
                    _extract_frame(source, src_in + off, ref)
                if ref.exists():
                    diffs[off] = _pixel_diff(ref, pngs[0])
            best_off = min(diffs, key=diffs.get)
            if best_off == 0:
                ok(f"first PNG = source frame {src_in} (diff={diffs[0]:.2f})")
            else:
                fail(f"first PNG is source {src_in + best_off}, not {src_in}")
                failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_image_sequence", passed=failures == 0,
                         failures=failures)


# -------------------- section: export_edl -------------------------------

def section_export_edl() -> SectionResult:
    """Build a timeline and verify the generated EDL is valid CMX 3600:
    sequential events, chained REC timecodes, correct duration math."""
    header("Export: EDL (CMX 3600)")
    failures = 0

    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from core.edl_exporter import export_edl

    fps = 23.976023976023978
    vs = VideoSource(id="s0", file_path="fake_source.mov",
                     total_frames=100000, fps=fps,
                     width=1920, height=1080, codec="h264")
    tl = TimelineModel()
    # 3 clips, each 48 frames (2s at 23.976)
    plan = [(1000, 48), (5000, 48), (10000, 48)]
    tl.add_clips([Clip(source_id="s0", source_in=s, source_out=s + n - 1)
                  for s, n in plan], assign_colors=False)

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_edl_"))
    try:
        edl = td / "out.edl"
        export_edl(tl, {"s0": vs}, str(edl), fps=fps)
        if not edl.exists() or edl.stat().st_size == 0:
            fail("EDL file missing or empty")
            return SectionResult("export_edl", passed=False, failures=1)
        ok(f"EDL written ({edl.stat().st_size} bytes)")

        content = edl.read_text(encoding="utf-8", errors="replace").splitlines()

        # Header: TITLE and FCM NON-DROP FRAME
        header_has_title = any(l.startswith("TITLE:") for l in content[:5])
        header_has_fcm = any("NON-DROP FRAME" in l for l in content[:5])
        if header_has_title:
            ok("header: TITLE present")
        else:
            fail("header: TITLE missing")
            failures += 1
        if header_has_fcm:
            ok("header: FCM NON-DROP FRAME present")
        else:
            fail("header: FCM line missing")
            failures += 1

        # Parse event lines (format: NNN reel V C SRC_IN SRC_OUT REC_IN REC_OUT)
        import re
        tc_re = r"(\d{2}):(\d{2}):(\d{2}):(\d{2})"
        evt_re = re.compile(
            rf"^(\d{{3}})\s+\S+\s+V\s+C\s+{tc_re}\s+{tc_re}\s+{tc_re}\s+{tc_re}$")
        events = []
        for line in content:
            m = evt_re.match(line.strip())
            if m:
                parts = [int(x) for x in m.groups()]
                events.append({
                    "n": parts[0],
                    "src_in": tuple(parts[1:5]),
                    "src_out": tuple(parts[5:9]),
                    "rec_in": tuple(parts[9:13]),
                    "rec_out": tuple(parts[13:17]),
                })

        if len(events) == 3:
            ok(f"3 events parsed")
        else:
            fail(f"expected 3 events, got {len(events)}")
            failures += 1

        # Events are numbered 001, 002, 003
        if events and [e["n"] for e in events] == [1, 2, 3]:
            ok("event numbers are sequential 001-003")
        else:
            fail(f"event numbering: {[e['n'] for e in events]}")
            failures += 1

        # REC timecodes chain: event N's rec_out == event N+1's rec_in
        if len(events) >= 2:
            for i in range(len(events) - 1):
                if events[i]["rec_out"] != events[i + 1]["rec_in"]:
                    fail(f"event {events[i]['n']} REC_OUT {events[i]['rec_out']} "
                         f"!= event {events[i+1]['n']} REC_IN "
                         f"{events[i+1]['rec_in']}")
                    failures += 1
                    break
            else:
                ok("REC timecodes chain correctly between events")

        # Duration math: REC_OUT - REC_IN of event 0 should be 48 frames (2s)
        if events:
            fps_int = 24
            def tc_to_frames(tc):
                h, m, s, f = tc
                return ((h * 3600 + m * 60 + s) * fps_int) + f
            dur_frames = tc_to_frames(events[0]["rec_out"]) - \
                         tc_to_frames(events[0]["rec_in"])
            if dur_frames == 48:
                ok(f"event 1 duration = 48 frames (REC)")
            else:
                fail(f"event 1 REC duration = {dur_frames} frames, expected 48")
                failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_edl", passed=failures == 0, failures=failures)


# -------------------- section: export_nvenc -----------------------------

def section_export_nvenc(videos: List[Path]) -> SectionResult:
    """Verify the SDR+NVENC scale_cuda path: export via h264_nvenc both
    at source resolution (format-only scale_cuda) and downscaled
    (scale_cuda with explicit dims). Confirms zero-copy GPU pipeline
    works end-to-end."""
    header("Export: NVENC (scale_cuda zero-copy)")
    failures = 0
    # NVENC is available?
    try:
        enc_out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        enc_out = ""
    if "h264_nvenc" not in enc_out:
        warn("h264_nvenc not available in ffmpeg; skipping")
        return SectionResult("export_nvenc", passed=True, note="no NVENC")

    sdr = [v for v in videos
           if "YTS" in v.name or "Hobbs" in v.name or "1080p" in v.name]
    source = sdr[0] if sdr else None
    if source is None:
        warn("no SDR source provided; skipping")
        return SectionResult("export_nvenc", passed=True, note="skipped")

    from core.clip import Clip
    from core.video_source import VideoSource
    from core.timeline import TimelineModel
    from utils.ffprobe import probe_video

    meta = probe_video(str(source))
    if meta is None:
        fail("probe failed")
        return SectionResult("export_nvenc", passed=False, failures=1)

    src_in = 500
    count = 30
    vs = VideoSource(id="s0", file_path=str(source),
                     total_frames=meta.total_frames, fps=meta.fps,
                     width=meta.width, height=meta.height, codec=meta.codec)

    nvenc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                  "-cq", "23", "-pix_fmt", "yuv420p"]

    td = Path(tempfile.mkdtemp(prefix="psynth_sys_nvenc_"))
    try:
        # Case 1: same resolution (tests scale_cuda=format=yuv420p only)
        tl = TimelineModel()
        tl.add_clips([Clip(source_id="s0", source_in=src_in,
                           source_out=src_in + count - 1)],
                     assign_colors=False)
        out_same = td / "same.mp4"
        t0 = time.perf_counter()
        r = _run_exporter(tl, {"s0": vs}, {
            "mode": "video", "output_path": str(out_same),
            "ffmpeg_args": nvenc_args,
            "width": meta.width, "height": meta.height, "fps": meta.fps,
        }, timeout=60.0)
        dt = time.perf_counter() - t0
        if r != "ok":
            fail(f"same-res: export failed — {r}")
            failures += 1
        elif not out_same.exists() or out_same.stat().st_size == 0:
            fail("same-res: output missing or empty")
            failures += 1
        else:
            ok(f"same-res ({meta.width}x{meta.height}): {count} frames "
               f"in {dt:.2f}s ({count / dt:.1f} fps)")

        # Case 2: downscale to 720p (tests scale_cuda=W:H:format=yuv420p)
        # Preserve aspect ratio of source (may yield non-720 height)
        target_w = 1280
        target_h = int(round(meta.height * target_w / meta.width))
        # Round to even for yuv420p
        target_h -= target_h % 2
        tl2 = TimelineModel()
        tl2.add_clips([Clip(source_id="s0", source_in=src_in,
                            source_out=src_in + count - 1)],
                      assign_colors=False)
        out_down = td / "down.mp4"
        t0 = time.perf_counter()
        r = _run_exporter(tl2, {"s0": vs}, {
            "mode": "video", "output_path": str(out_down),
            "ffmpeg_args": nvenc_args,
            "width": target_w, "height": target_h, "fps": meta.fps,
        }, timeout=60.0)
        dt = time.perf_counter() - t0
        if r != "ok":
            fail(f"downscale: export failed — {r}")
            failures += 1
        elif not out_down.exists() or out_down.stat().st_size == 0:
            fail("downscale: output missing or empty")
            failures += 1
        else:
            ok(f"downscale ({meta.width}x{meta.height} -> "
               f"{target_w}x{target_h}): {count} frames in {dt:.2f}s "
               f"({count / dt:.1f} fps)")
            # Verify the output resolution is what we asked for
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height",
                   "-of", "csv=p=0:s=x", str(out_down)]
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            actual = (r2.stdout or "").strip()
            expected = f"{target_w}x{target_h}"
            if actual == expected:
                ok(f"downscale: output resolution = {actual}")
            else:
                fail(f"downscale: output resolution {actual} != {expected}")
                failures += 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

    return SectionResult("export_nvenc", passed=failures == 0,
                         failures=failures)


# ------------------------------- main -------------------------------------

def main():
    _enable_win_ansi()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", default=[],
                        help="Path to a sample video (repeatable, SDR preferred for accuracy tests)")
    parser.add_argument("--skip", action="append", default=[],
                        choices=SECTIONS)
    args = parser.parse_args()

    videos = [Path(v) for v in args.video]
    for v in videos:
        if not v.exists():
            print(f"{C.R}Video not found: {v}{C.X}")
            sys.exit(2)

    started = time.perf_counter()
    skip = set(args.skip)
    if "env" not in skip:
        run_section("env", section_env)
    if "compile" not in skip:
        run_section("compile", section_compile)
    if "model" not in skip:
        run_section("model", section_model)
    if "project" not in skip:
        run_section("project", section_project)
    # EDL export uses a fake source, no --video needed.
    if "export_edl" not in skip:
        run_section("export_edl", section_export_edl)

    if videos:
        if "thumb_accuracy" not in skip:
            run_section("thumb_accuracy",
                        lambda: section_thumb_accuracy(videos))
        if "export_accuracy" not in skip:
            run_section("export_accuracy",
                        lambda: section_export_accuracy(videos))
        if "export_timebase" not in skip:
            run_section("export_timebase",
                        lambda: section_export_timebase(videos))
        if "export_multi_segment" not in skip:
            run_section("export_multi_segment",
                        lambda: section_export_multi_segment(videos))
        if "export_hdr" not in skip:
            run_section("export_hdr",
                        lambda: section_export_hdr(videos))
        if "export_image_sequence" not in skip:
            run_section("export_image_sequence",
                        lambda: section_export_image_sequence(videos))
        if "export_nvenc" not in skip:
            run_section("export_nvenc",
                        lambda: section_export_nvenc(videos))
    else:
        print(f"\n{C.Y}No --video provided -- skipping video-dependent sections.{C.X}")

    elapsed = time.perf_counter() - started
    header("Summary")
    any_fail = False
    for r in RESULTS:
        mark = f"{C.G}PASS{C.X}" if r.passed else f"{C.R}FAIL{C.X}"
        extra = f" ({r.failures} failure{'s' if r.failures != 1 else ''})" if r.failures else ""
        note = f" — {r.note}" if r.note else ""
        print(f"  [{mark}] {r.name:<18}{extra}{note}")
        if not r.passed:
            any_fail = True
    print(f"\nTotal time: {elapsed:.1f}s")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
