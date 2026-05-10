"""Audio-drift diagnostic harness for PrismaSynth video exports.

Loads a real ``.psynth`` project, drives an end-to-end ProRes-MOV export
with embedded PCM audio through the real :class:`core.exporter.Exporter`,
then ffprobes the output to compare total audio sample count against the
sum of per-segment expected sample counts. Reports drift in samples and
milliseconds.

The point: per-segment audio drift in the export pipeline is a numerical
question, not an eyeball question. The harness gives a reproducible
number we can drive a fix against.

Usage::

    venv\\Scripts\\python scripts\\audio_drift_check.py PROJECT.psynth
        [--group NAME]          People-group filter; default 'billycrystal'
                                if present, else no filter.
        [--codec KEY]           Codec preset key; default 'prores_proxy'.
        [--out DIR]             Artefact directory; default %TEMP%\\audio_drift_check_*.
        [--prefix-frames N]     Cap output to first N timeline frames via in/out
                                range (fast iteration). Default: full timeline.

Exit code is 0 when drift is below 1 ms total, non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ["PATH"] = str(SRC) + os.pathsep + os.environ.get("PATH", "")


# Per-codec preset table, mirroring src/ui/export_dialog.py::VIDEO_PRESETS
# (kept narrow — the harness only needs ProRes/MOV right now).
_PRESETS = {
    "prores_proxy": {
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "0", "-pix_fmt", "yuv422p10le"],
    },
    "prores_lt": {
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "1", "-pix_fmt", "yuv422p10le"],
    },
    "prores_standard": {
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "2", "-pix_fmt", "yuv422p10le"],
    },
    "prores_hq": {
        "ext": ".mov",
        "args": ["-c:v", "prores_aw", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    },
}


def _ffprobe_audio(path: Path) -> dict:
    """Return audio-stream metadata from ``path``: nb_samples, sample_rate,
    duration_ts, time_base, codec_name. Raises on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=nb_samples,sample_rate,duration_ts,time_base,codec_name,channels,duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No audio stream in {path}")
    return data["streams"][0]


def _ffprobe_video(path: Path) -> dict:
    """Return video-stream metadata from ``path``: duration, nb_frames,
    avg_frame_rate, r_frame_rate."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_frames,duration,avg_frame_rate,r_frame_rate,codec_name",
        "-of", "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream in {path}")
    return data["streams"][0]


def _expected_samples_for_segments(segments, sample_rate: int = 48000) -> int:
    """Sum per-segment expected audio sample counts. For NTSC sources at
    23.976 fps, each video frame at 48kHz is exactly 2002 samples
    (48000 * 1001 / 24000) — but compute the rational directly to avoid
    float drift. ``segments`` is whatever ``_build_segments`` returned:
    tuples of (path, src_in, count, src_fps, source_id)."""
    total = 0
    for path, _src_in, count, src_fps, _sid in segments:
        # Rational sample count: count frames * (sample_rate / src_fps).
        # Using a tolerant rounding step — for clean rates this is exact.
        # NTSC 23.976 = 24000/1001, samples per frame = 48000*1001/24000 = 2002.
        if abs(src_fps - 24000.0 / 1001.0) < 1e-3:
            samples = count * sample_rate * 1001 // 24000
        elif abs(src_fps - 30000.0 / 1001.0) < 1e-3:
            samples = count * sample_rate * 1001 // 30000
        elif abs(src_fps - 60000.0 / 1001.0) < 1e-3:
            samples = count * sample_rate * 1001 // 60000
        else:
            # Integer rates (24, 25, 30, 50, 60) divide cleanly. Fall back
            # to a round() — float error is sub-sample.
            samples = round(count * sample_rate / src_fps)
        total += samples
    return total


def _resolve_group_filter(timeline, name: Optional[str]) -> Optional[dict]:
    """Look up a People group by display name; build the filter dict the
    Exporter expects. ``name=None`` returns no filter."""
    if not name:
        return None
    for gid, g in timeline.groups.items():
        if g.name.casefold() == name.casefold():
            return {"group_ids": [gid], "include_untagged": False}
    available = ", ".join(g.name for g in timeline.groups.values()) or "(none)"
    raise SystemExit(
        f"Group {name!r} not found in project. Available: {available}")


def _build_timeline(loaded: dict):
    """Materialise a TimelineModel from a load_project() result. Mirrors
    MainWindow._load_from but stripped to just timeline state."""
    from core.timeline import TimelineModel
    tl = TimelineModel()
    tl.set_groups_bulk(loaded.get("groups", {}).values())
    tl.add_clips(loaded["clips"], assign_colors=False)
    return tl


def _drive_export(timeline, sources: dict, settings: dict,
                  timeout_s: float = 1800.0) -> str:
    """Run Exporter.export to completion (or timeout). Mirrors the
    QEventLoop pattern in scripts/system_check.py::_export_via_exporter."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from PySide6.QtCore import Qt as _Qt
    from core.exporter import Exporter

    _ = QCoreApplication.instance() or QCoreApplication(sys.argv)
    exporter = Exporter(timeline, sources)

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

    def on_status(msg):
        # Cheap progress indicator on stderr so the user sees the export
        # is doing something during the multi-minute run.
        sys.stderr.write(f"  status: {msg}\n")
        sys.stderr.flush()

    exporter.finished.connect(on_finished, _Qt.ConnectionType.QueuedConnection)
    exporter.error.connect(on_error, _Qt.ConnectionType.QueuedConnection)
    exporter.cancelled.connect(on_cancelled, _Qt.ConnectionType.QueuedConnection)
    exporter.status.connect(on_status, _Qt.ConnectionType.QueuedConnection)

    exporter.export(settings)
    QTimer.singleShot(int(timeout_s * 1000),
                      lambda: (exporter.cancel(), loop.quit()))
    loop.exec()

    if result["kind"] == "ok":
        return "ok"
    if result["kind"] == "error":
        return f"error:{result['msg'][:200]}"
    if result["kind"] == "cancelled":
        return "timeout-or-cancelled"
    return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project", type=Path, help=".psynth project path")
    p.add_argument("--group", default="billycrystal",
                   help="People-group filter; pass an empty string to "
                        "disable. Default: 'billycrystal'.")
    p.add_argument("--codec", default="prores_proxy", choices=list(_PRESETS),
                   help="Codec preset (default: prores_proxy)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: %%TEMP%%/audio_drift_check_<ts>)")
    p.add_argument("--prefix-frames", type=int, default=0,
                   help="Cap output to first N timeline frames (default: full)")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="Export timeout in seconds (default: 1800)")
    args = p.parse_args(argv)

    if not args.project.exists():
        print(f"Project not found: {args.project}", file=sys.stderr)
        return 2

    if shutil.which("ffprobe") is None:
        print("ffprobe not on PATH", file=sys.stderr)
        return 2

    out_dir = args.out
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(tempfile.gettempdir()) / f"audio_drift_check_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[harness] Artefacts: {out_dir}")

    # ---- Load project ----------------------------------------------------
    from core.project import load_project
    loaded = load_project(str(args.project))
    sources = loaded["sources"]
    if not sources:
        print("Project has no sources.", file=sys.stderr)
        return 2

    # Use the first source as the timeline reference.
    ref_src = next(iter(sources.values()))
    width, height, fps = ref_src.width, ref_src.height, ref_src.fps
    print(f"[harness] Reference source: {ref_src.file_path}")
    print(f"[harness]   {width}x{height}@{fps} ({ref_src.codec}), "
          f"audio {ref_src.audio_codec or '(none)'} "
          f"{ref_src.audio_sample_rate}Hz x{ref_src.audio_channels}")
    if ref_src.audio_sample_rate != 48000:
        print(f"[harness] Source sample rate is {ref_src.audio_sample_rate}, "
              f"not 48000 — expected-sample math assumes 48kHz output.")

    timeline = _build_timeline(loaded)
    print(f"[harness] Loaded {len(timeline.clips)} clips, "
          f"{len(timeline.groups)} groups")

    # ---- Optional prefix truncation via in/out ---------------------------
    if args.prefix_frames > 0:
        total = timeline.get_total_duration_frames()
        end = min(args.prefix_frames - 1, total - 1)
        timeline.set_in_point(0)
        timeline.set_out_point(end)
        use_range = True
        print(f"[harness] Truncating to first {end + 1} timeline frames "
              f"via in/out range")
    else:
        use_range = False

    # ---- Group filter ----------------------------------------------------
    group_name = args.group.strip() or None
    group_filter = _resolve_group_filter(timeline, group_name)
    print(f"[harness] Group filter: {group_name or '(none)'} -> "
          f"{group_filter}")

    # ---- Compute expected samples ---------------------------------------
    # Drive _build_segments with the same group filter / range so the
    # expected sample count matches what the exporter actually emits.
    from core.exporter import Exporter as _ExpForSeg
    seg_probe = _ExpForSeg(timeline, sources)
    seg_probe._group_filter = group_filter
    seg_probe._use_render_range = use_range
    seg_probe._include_gaps = False
    seg_probe._export_fps = fps
    segments = seg_probe._build_segments()
    n_segments = len(segments)
    total_video_frames = sum(s[2] for s in segments)
    expected_samples = _expected_samples_for_segments(segments, sample_rate=48000)
    expected_seconds = expected_samples / 48000.0
    print(f"[harness] Surviving segments: {n_segments}")
    print(f"[harness]   total video frames: {total_video_frames}")
    print(f"[harness]   expected audio samples @48kHz: "
          f"{expected_samples} ({expected_seconds:.3f}s)")
    if n_segments == 0:
        print("Nothing to export — group filter excluded everything.")
        return 2

    # ---- Build settings dict --------------------------------------------
    preset = _PRESETS[args.codec]
    output_path = out_dir / f"export{preset['ext']}"
    settings = {
        "mode": "video",
        "output_path": str(output_path),
        "codec_key": args.codec,
        "ffmpeg_args": list(preset["args"]),
        "ext": preset["ext"],
        "width": width,
        "height": height,
        "fps": fps,
        "audio_mode": "embedded",
        "group_filter": group_filter,
        "include_gaps": False,
        "use_render_range": use_range,
    }
    print(f"[harness] Output: {output_path}")
    print(f"[harness] Codec: {args.codec} ({' '.join(preset['args'])})")

    # ---- Run export ------------------------------------------------------
    t0 = time.time()
    result = _drive_export(timeline, sources, settings, timeout_s=args.timeout)
    elapsed = time.time() - t0
    print(f"[harness] Export result: {result} ({elapsed:.1f}s)")
    if result != "ok":
        return 3
    if not output_path.exists():
        print("Export reported OK but output file is missing.", file=sys.stderr)
        return 3

    # ---- Probe output ----------------------------------------------------
    aprobe = _ffprobe_audio(output_path)
    vprobe = _ffprobe_video(output_path)
    print(f"[harness] Output audio: {aprobe}")
    print(f"[harness] Output video: {vprobe}")

    actual_samples = int(aprobe.get("nb_samples") or 0)
    if actual_samples == 0:
        # Fallback: derive from duration + sample_rate
        sr = int(aprobe["sample_rate"])
        dur = float(aprobe.get("duration") or 0.0)
        actual_samples = round(dur * sr)
        print(f"[harness] nb_samples missing - derived "
              f"{actual_samples} from duration {dur}s")

    # Extract the audio stream to raw PCM s16le to count samples by raw byte
    # count - bypasses any container metadata (edts/elst atoms) that might
    # report a shorter playback duration than what's actually stored.
    raw_path = out_dir / "raw_audio.s16le"
    extract_cmd = [
        "ffmpeg", "-y", "-nostdin", "-v", "error",
        "-i", str(output_path),
        "-vn", "-c:a", "pcm_s16le", "-f", "s16le",
        "-ac", "2", "-ar", "48000",
        str(raw_path),
    ]
    subprocess.run(extract_cmd, check=True)
    raw_bytes = raw_path.stat().st_size
    raw_samples = raw_bytes // 4  # 2 bytes per sample x 2 channels
    print(f"[harness] Raw PCM extract: {raw_bytes} bytes -> "
          f"{raw_samples} samples")
    if raw_samples != actual_samples:
        print(f"[harness]   container reports {actual_samples} samples but "
              f"raw stream contains {raw_samples} - "
              f"diff {raw_samples - actual_samples} (likely edts/elst)")
        actual_samples = raw_samples

    drift_samples = actual_samples - expected_samples
    drift_ms = drift_samples / 48.0  # 48 samples per ms at 48kHz

    # Video-side sanity check
    actual_video_frames = int(vprobe.get("nb_frames") or 0)
    video_frame_drift = actual_video_frames - total_video_frames

    # ---- Report ---------------------------------------------------------
    report_lines = [
        "Audio drift check — report",
        "=" * 70,
        f"Project:   {args.project}",
        f"Group:     {group_name}",
        f"Codec:     {args.codec}",
        f"Output:    {output_path}",
        f"Elapsed:   {elapsed:.1f}s",
        "",
        f"Segments:        {n_segments}",
        f"Video frames:    expected={total_video_frames}, "
        f"actual={actual_video_frames}, drift={video_frame_drift}",
        f"Audio samples:   expected={expected_samples}, "
        f"actual={actual_samples}, drift={drift_samples}",
        f"Audio drift ms:  {drift_ms:+.3f} ms",
        f"Per-segment avg: {drift_ms / max(1, n_segments):+.4f} ms/segment",
        "",
        f"Verdict: {'PASS' if abs(drift_ms) < 1.0 else 'FAIL'} "
        f"(threshold: 1 ms total)",
    ]
    report = "\n".join(report_lines)
    print()
    print(report)

    report_path = out_dir / "drift_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[harness] Report written: {report_path}")

    return 0 if abs(drift_ms) < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
