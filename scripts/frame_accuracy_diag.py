"""Frame-accuracy diagnostic for FCPXML / OTIO exporters.

Runs both exporters against a real .psynth project + source file, parses the
emitted seek values back, and reports — per clip — what frame each plausible
Resolve reader model would actually decode. Cross-references the predictions
against the source's real PTS table from ffprobe so source-side jitter shows
up grounded in fact rather than guessed at.

This is a diagnostic, not a fix. Run it against a project that misaligns on
re-import to Resolve. The output table tells you which assumption is wrong:

  * a model column flagging mismatch -> Resolve's reader behaves like that
    model on your machine; the corresponding nudge needs adjustment.
  * pts_delta column flagging mismatch -> source PTS is jittery / VFR /
    has gaps; no XML/OTIO trick can fix it on its own.
  * recovered_N != intended_N -> the writer or this harness has drifted
    from spec (fail loud).

Usage:
  venv\\Scripts\\python scripts\\frame_accuracy_diag.py
        --project path\\to\\test.psynth
        --source  path\\to\\source.mp4
        [--include-gaps] [--use-render-range]
        [--fps 23.976023976023978] [--out-dir DIR] [--verbose]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.clip import Clip
from core.timeline import TimelineModel
from core.video_source import VideoSource
from core.project import load_project
from core.xml_exporter import export_fcpxml, _rate_to_frame_duration
from core.otio_exporter import export_otio, _SRC_SEEK_NUDGE


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


def header(t): print(f"\n{C.B}{C.CYAN}{'='*78}\n{t}\n{'='*78}{C.X}")
def ok(m): print(f"  {C.G}[OK]{C.X}   {m}")
def fail(m): print(f"  {C.R}[FAIL]{C.X} {m}")
def warn(m): print(f"  {C.Y}[WARN]{C.X} {m}")
def info(m): print(f"         {m}")


# ----------------------------- ffprobe helpers ----------------------------

@dataclass
class StreamInfo:
    time_base: str
    r_frame_rate: str
    avg_frame_rate: str
    nb_frames: int
    start_pts: int
    start_time: float
    pts_times: List[float]


def _probe_stream(path: str) -> StreamInfo:
    """Probe a single video stream's timing data + per-packet pts table."""
    s = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=True, timeout=60,
    )
    sj = json.loads(s.stdout)["streams"][0]
    nb_frames = int(sj.get("nb_frames", 0)) if sj.get("nb_frames") else 0
    start_pts = int(sj.get("start_pts", 0) or 0)
    try:
        start_time = float(sj.get("start_time", 0) or 0)
    except (TypeError, ValueError):
        start_time = 0.0

    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time",
         "-of", "csv=print_section=0", path],
        capture_output=True, text=True, check=True, timeout=300,
    )
    pts_times = []
    for line in p.stdout.splitlines():
        line = line.strip().rstrip(",")
        if not line or line == "N/A":
            continue
        try:
            pts_times.append(float(line))
        except ValueError:
            continue
    pts_times.sort()

    return StreamInfo(
        time_base=sj.get("time_base", ""),
        r_frame_rate=sj.get("r_frame_rate", ""),
        avg_frame_rate=sj.get("avg_frame_rate", ""),
        nb_frames=nb_frames,
        start_pts=start_pts,
        start_time=start_time,
        pts_times=pts_times,
    )


# ------------------------- intended-output walker -------------------------

@dataclass
class Intended:
    """Ground truth produced by re-walking the timeline with the same loop
    shape the exporters use (xml_exporter.py:266-320, otio_exporter.py:174-216).
    """
    clip_idx: int
    source_id: str
    intended_frame: int   # source-domain frame the writer aimed at
    eff_duration: int
    packed_offset: int    # cumulative non-gap timeline offset


def _compute_intended(timeline: TimelineModel, sources: Dict[str, VideoSource],
                      include_gaps: bool, use_render_range: bool
                      ) -> List[Intended]:
    clips = timeline.clips
    if use_render_range:
        render_start, render_end = timeline.get_render_range()
    else:
        render_start = 0
        render_end = timeline.get_total_duration_frames() - 1

    out: List[Intended] = []
    timeline_pos = 0
    rec_pos_frames = 0  # packed-layout offset (mirrors xml_exporter.py:267)
    idx = 0

    for clip in clips:
        clip_start = timeline_pos
        clip_end = timeline_pos + clip.duration_frames - 1
        timeline_pos += clip.duration_frames

        if clip_end < render_start or clip_start > render_end:
            continue

        eff_start = max(clip_start, render_start)
        eff_end = min(clip_end, render_end)
        eff_duration = eff_end - eff_start + 1

        if clip.is_gap:
            # Gaps don't appear in the asset-clip parse list when
            # include_gaps=False, so we skip without bumping idx.
            # When include_gaps=True the writer emits a <gap> / Gap.1
            # but those aren't asset-clips either — caller filters them.
            continue

        if clip.source_id not in sources:
            continue

        offset_in_clip = eff_start - clip_start
        src_in_frame = clip.source_in + offset_in_clip
        spine_offset = (eff_start - render_start) if include_gaps else rec_pos_frames

        out.append(Intended(
            clip_idx=idx,
            source_id=clip.source_id,
            intended_frame=src_in_frame,
            eff_duration=eff_duration,
            packed_offset=spine_offset,
        ))
        idx += 1
        rec_pos_frames += eff_duration

    return out


# ----------------------------- FCPXML parsing -----------------------------

@dataclass
class XmlClipRow:
    num: int
    den: int
    dur_num: int
    offset_num: int
    raw_start: str


def _parse_fraction(s: str) -> Tuple[int, int]:
    s = s.strip()
    if s == "0s":
        return (0, 1)
    if not s.endswith("s"):
        raise ValueError(f"unexpected time string: {s!r}")
    body = s[:-1]
    if "/" in body:
        a, b = body.split("/", 1)
        return (int(a), int(b))
    # bare integer seconds
    return (int(body), 1)


def _parse_fcpxml(path: str) -> List[XmlClipRow]:
    tree = ET.parse(path)
    root = tree.getroot()
    spine = root.find("library/event/project/sequence/spine")
    if spine is None:
        raise RuntimeError("no spine in FCPXML")
    rows: List[XmlClipRow] = []
    for ac in spine.findall("asset-clip"):
        s = ac.get("start", "0s")
        d = ac.get("duration", "0s")
        o = ac.get("offset", "0s")
        s_num, s_den = _parse_fraction(s)
        d_num, _ = _parse_fraction(d)
        o_num, _ = _parse_fraction(o)
        rows.append(XmlClipRow(
            num=s_num, den=s_den, dur_num=d_num,
            offset_num=o_num, raw_start=s,
        ))
    return rows


# ------------------------------ OTIO parsing ------------------------------

@dataclass
class OtioClipRow:
    value: float
    rate: float
    duration: float


def _parse_otio(path: str) -> Tuple[List[OtioClipRow], List[dict], dict]:
    """Returns (clip_rows_in_track_order, gap_items, first_clip_avail_range).

    Track items are filtered to Clip.2 only for the main list; gaps are
    returned separately so caller can verify their integer source_range.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("OTIO_SCHEMA") != "Timeline.1":
        raise RuntimeError(f"unexpected OTIO schema: {doc.get('OTIO_SCHEMA')}")
    stack = doc["tracks"]
    track = stack["children"][0]
    clip_rows: List[OtioClipRow] = []
    gaps: List[dict] = []
    first_avail: dict = {}
    for child in track["children"]:
        sch = child.get("OTIO_SCHEMA", "")
        if sch.startswith("Clip"):
            sr = child["source_range"]
            clip_rows.append(OtioClipRow(
                value=float(sr["start_time"]["value"]),
                rate=float(sr["start_time"]["rate"]),
                duration=float(sr["duration"]["value"]),
            ))
            if not first_avail:
                ref = child["media_references"]["DEFAULT_MEDIA"]
                first_avail = ref.get("available_range", {})
        elif sch.startswith("Gap"):
            gaps.append(child)
    return clip_rows, gaps, first_avail


# -------------------------- reader model simulation -----------------------

@dataclass
class ModelResult:
    name: str
    predicted: int


def _xml_models(num: int, den: int, source_fps: float, frame_num: int,
                frame_den: int, is_ntsc: bool) -> List[ModelResult]:
    if num == 0:
        # writer bypasses nudge for frame=0 (xml_exporter.py:97-98); models
        # all collapse to 0. Return a single placeholder so the table stays
        # uniform.
        return [ModelResult(n, 0) for n in
                ("M1_floor", "M2_round", "M3_trunc")]
    t = num / den
    out = [
        ModelResult("M1_floor", math.floor(t * source_fps)),
        ModelResult("M2_round", int(round(t * source_fps))),
        ModelResult("M3_trunc", math.trunc(t * source_fps)),
    ]
    if is_ntsc:
        # Resolve's documented internal rewrite to 1/24-tick storage.
        # See xml_exporter._src_seek_str rationale (lines 78-91).
        t24 = round((num / den) * 24) / 24.0
        out.append(ModelResult("M4_24tick_floor",
                               math.floor(t24 * source_fps)))
        out.append(ModelResult("M5_24tick_round",
                               int(round(t24 * source_fps))))
        # M6: declared rate from frameDuration (often equals source_fps but
        # not always — surfaces format-vs-source rate mismatch). frameDuration
        # is seconds-per-frame so the rate is its reciprocal.
        decl_fps = frame_den / frame_num
        out.append(ModelResult("M6_decl_floor",
                               math.floor(t * decl_fps)))
    return out


def _otio_models(value: float, rate: float, source_fps: float,
                 is_ntsc: bool) -> List[ModelResult]:
    t = value / rate
    out = [
        ModelResult("O1_floor", math.floor(t * source_fps)),
        ModelResult("O2_round", int(round(t * source_fps))),
        ModelResult("O3_trunc", math.trunc(t * source_fps)),
    ]
    if is_ntsc:
        # Resolve's reported 23.976023976023979 vs our 23.976023976023978 —
        # the ULP drift the +0.25 nudge is designed to defeat
        # (otio_exporter.py:38-45).
        rrate = 23.976023976023979
        out.append(ModelResult("O4_floor_rrate",
                               math.floor(t * rrate)))
        out.append(ModelResult("O5_round_rrate",
                               int(round(t * rrate))))
        out.append(ModelResult("O6_trunc_rrate",
                               math.trunc(t * rrate)))
    return out


# ----------------------- PTS bracket cross-reference ----------------------

def _pts_bracket(pts_times: List[float], seek_seconds: float,
                 start_time: float) -> int:
    """Return the index k such that pts_times[k] is the latest PTS <= seek.
    Subtracts container start_time from seek before bracketing (TS files,
    some MOVs use a non-zero PTS origin).
    """
    if not pts_times:
        return -1
    adj = seek_seconds - start_time
    k = bisect.bisect_right(pts_times, adj) - 1
    return max(0, k)


# ------------------------------ main analysis -----------------------------

def _is_ntsc(fps: float) -> bool:
    return any(abs(fps - r) < 0.01 for r in (24000/1001, 30000/1001, 60000/1001))


def _swap_source_path(sources: Dict[str, VideoSource], new_path: str) -> Optional[str]:
    """Swap any source whose basename matches new_path's basename. Returns
    the source_id that was swapped, or None if no match."""
    target = os.path.basename(new_path).lower()
    for sid, s in sources.items():
        if os.path.basename(s.file_path).lower() == target:
            s.file_path = new_path
            return sid
    # fallback: if there's only one source, swap it regardless
    if len(sources) == 1:
        sid = next(iter(sources))
        sources[sid].file_path = new_path
        return sid
    return None


def main():
    _enable_win_ansi()

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project", required=True, help="Path to .psynth file")
    ap.add_argument("--source", required=True,
                    help="Path to the source video file to probe")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the temp .fcpxml/.otio (default: tmp)")
    ap.add_argument("--include-gaps", action="store_true")
    ap.add_argument("--use-render-range", action="store_true")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override sequence fps (default: first source's fps)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(
        prefix="psynth_diag_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load project & build timeline ----------------------------------
    header("Load project & re-point source")
    proj = load_project(args.project)
    sources: Dict[str, VideoSource] = proj["sources"]
    if not sources:
        fail("project has no sources")
        return 2
    swapped = _swap_source_path(sources, args.source)
    if swapped is None:
        warn(f"no source basename matched {os.path.basename(args.source)}; "
             f"PTS cross-reference will use the project's stored paths")
    else:
        ok(f"source {swapped} -> {args.source}")

    tl = TimelineModel()
    tl.add_clips(proj["clips"], assign_colors=False)
    if args.use_render_range:
        if proj["in_point"] is not None:
            tl.set_in_point(proj["in_point"])
        if proj["out_point"] is not None:
            tl.set_out_point(proj["out_point"])
        info(f"render range = {tl.get_render_range()}")

    fps = args.fps if args.fps is not None else next(iter(sources.values())).fps
    is_ntsc = _is_ntsc(fps)
    frame_num, frame_den = _rate_to_frame_duration(fps)
    info(f"sequence fps = {fps}  (NTSC fractional: {is_ntsc})")
    info(f"frame duration = {frame_num}/{frame_den}s")

    # --- Run both exporters ---------------------------------------------
    xml_path = out_dir / "diag.fcpxml"
    otio_path = out_dir / "diag.otio"
    export_fcpxml(tl, sources, str(xml_path),
                  include_gaps=args.include_gaps,
                  use_render_range=args.use_render_range, fps=fps)
    export_otio(tl, sources, str(otio_path),
                include_gaps=args.include_gaps,
                use_render_range=args.use_render_range, fps=fps)
    if not xml_path.exists() or not otio_path.exists():
        fail("one of the exporters did not produce output")
        return 2
    ok(f"FCPXML written: {xml_path} ({xml_path.stat().st_size} bytes)")
    ok(f"OTIO   written: {otio_path} ({otio_path.stat().st_size} bytes)")

    # --- Build intended ground-truth list --------------------------------
    intended = _compute_intended(tl, sources, args.include_gaps,
                                 args.use_render_range)
    if not intended:
        fail("no clips emitted (empty render range?)")
        return 2
    info(f"intended clip count: {len(intended)}")

    # --- Parse outputs ---------------------------------------------------
    xml_rows = _parse_fcpxml(str(xml_path))
    otio_rows, otio_gaps, first_avail = _parse_otio(str(otio_path))
    if len(xml_rows) != len(intended):
        fail(f"FCPXML asset-clip count {len(xml_rows)} != intended "
             f"{len(intended)}")
        return 2
    if len(otio_rows) != len(intended):
        fail(f"OTIO Clip.2 count {len(otio_rows)} != intended "
             f"{len(intended)}")
        return 2
    ok(f"parsed {len(xml_rows)} FCPXML asset-clips, {len(otio_rows)} OTIO clips")

    # --- available_range sanity (otio_exporter.py:89) -------------------
    if first_avail:
        ar_start = first_avail["start_time"]["value"]
        ar_dur = first_avail["duration"]["value"]
        first_intended = intended[0]
        first_src = sources[first_intended.source_id]
        if ar_start != 0.0:
            fail(f"OTIO available_range.start_time = {ar_start}, expected 0.0")
        elif ar_dur != float(first_src.total_frames):
            fail(f"OTIO available_range.duration = {ar_dur}, "
                 f"expected {first_src.total_frames}")
        else:
            ok("OTIO available_range integer & matches source.total_frames")

    # --- Gap source_range sanity ----------------------------------------
    if args.include_gaps and otio_gaps:
        bad = 0
        for g in otio_gaps:
            sr = g["source_range"]
            sv = sr["start_time"]["value"]
            dv = sr["duration"]["value"]
            if sv != 0.0 or not float(dv).is_integer():
                bad += 1
        if bad:
            fail(f"OTIO gaps with non-integer source_range: {bad}")
        else:
            ok(f"all {len(otio_gaps)} OTIO gaps integer & start=0 (no nudge)")

    # --- Probe source PTS (cache per source.id) -------------------------
    pts_cache: Dict[str, StreamInfo] = {}
    for sid in {it.source_id for it in intended}:
        path = sources[sid].file_path
        if not os.path.exists(path):
            warn(f"source path missing for {sid}: {path}")
            continue
        try:
            pts_cache[sid] = _probe_stream(path)
            si = pts_cache[sid]
            info(f"probed {sid}: time_base={si.time_base} "
                 f"r={si.r_frame_rate} avg={si.avg_frame_rate} "
                 f"nb_frames={si.nb_frames} start_time={si.start_time:.6f} "
                 f"pts_count={len(si.pts_times)}")
            if si.start_time > 1e-6:
                warn(f"  source has start_time > 0 ({si.start_time:.6f}s) — "
                     f"subtracted before bracketing")
            if si.pts_times:
                diffs = [si.pts_times[i+1] - si.pts_times[i]
                         for i in range(len(si.pts_times) - 1)]
                if diffs:
                    nominal = 1.0 / sources[sid].fps
                    info(f"  pts diff: min={min(diffs)*1000:.4f}ms "
                         f"max={max(diffs)*1000:.4f}ms "
                         f"nominal={nominal*1000:.4f}ms")
                    if max(diffs) > 1.5 * nominal:
                        warn(f"  source has gap or duplicate frame "
                             f"(max pts-diff > 1.5x nominal) — "
                             f"frame mapping will drift here regardless of XML")
            if si.nb_frames and si.nb_frames != sources[sid].total_frames:
                warn(f"  ffprobe nb_frames ({si.nb_frames}) != project "
                     f"total_frames ({sources[sid].total_frames})")
        except Exception as e:
            warn(f"ffprobe failed for {sid}: {e}")

    # --- Per-clip table --------------------------------------------------
    header("FCPXML asset-clip analysis")
    fcpxml_mismatches: List[Tuple[int, int, str, int]] = []
    for it, row in zip(intended, xml_rows):
        source_fps = sources[it.source_id].fps
        is_n = _is_ntsc(source_fps)
        # invert writer (xml_exporter._src_seek_str line 99)
        if row.num == 0:
            recovered = 0
        else:
            recovered = (row.num - frame_num // 2) // frame_num
        if recovered != it.intended_frame:
            fail(f"clip {it.clip_idx}: recovered {recovered} != "
                 f"intended {it.intended_frame} (raw start={row.raw_start}) "
                 f"— writer/diag drift; halt")
            return 2

        models = _xml_models(row.num, row.den, source_fps,
                             frame_num, frame_den, is_n)
        seek_t = (row.num / row.den) if row.num else 0.0
        si = pts_cache.get(it.source_id)
        if si and si.pts_times:
            pts_k = _pts_bracket(si.pts_times, seek_t, si.start_time)
            pts_delta = pts_k - it.intended_frame
        else:
            pts_k = -1
            pts_delta = 0  # neutral when no PTS data

        flagged = [m.name for m in models if m.predicted != it.intended_frame]
        if pts_k >= 0 and pts_delta != 0:
            flagged.append(f"pts({pts_delta:+d})")
        if flagged:
            fcpxml_mismatches.append((it.clip_idx, it.intended_frame,
                                      ",".join(flagged), pts_k))

        if args.verbose or flagged:
            mstr = " ".join(f"{m.name.split('_',1)[0]}={m.predicted}"
                            for m in models)
            tag = "MISMATCH" if flagged else "ok"
            print(f"  [{tag}] clip {it.clip_idx:3d} N={it.intended_frame:6d} "
                  f"start={row.raw_start:24s} dur={row.dur_num // frame_num:5d} "
                  f"off={row.offset_num // frame_num:6d} "
                  f"{mstr} pts_k={pts_k} delta={pts_delta:+d}")

    if not fcpxml_mismatches:
        ok("FCPXML: every emitted asset-clip frame-perfect under all models "
           "+ PTS bracket")
    else:
        fail(f"FCPXML: {len(fcpxml_mismatches)} clips have at least one "
             f"reader model or PTS bracket disagreeing")

    header("OTIO Clip.2 analysis")
    otio_mismatches: List[Tuple[int, int, str, int]] = []
    for it, row in zip(intended, otio_rows):
        source_fps = sources[it.source_id].fps
        is_n = _is_ntsc(source_fps)
        recovered = int(round(row.value - _SRC_SEEK_NUDGE))
        if recovered != it.intended_frame:
            fail(f"clip {it.clip_idx}: recovered {recovered} != "
                 f"intended {it.intended_frame} (value={row.value}) "
                 f"— writer/diag drift; halt")
            return 2

        models = _otio_models(row.value, row.rate, source_fps, is_n)
        seek_t = row.value / row.rate
        si = pts_cache.get(it.source_id)
        if si and si.pts_times:
            pts_k = _pts_bracket(si.pts_times, seek_t, si.start_time)
            pts_delta = pts_k - it.intended_frame
        else:
            pts_k = -1
            pts_delta = 0

        flagged = [m.name for m in models if m.predicted != it.intended_frame]
        if pts_k >= 0 and pts_delta != 0:
            flagged.append(f"pts({pts_delta:+d})")
        if flagged:
            otio_mismatches.append((it.clip_idx, it.intended_frame,
                                    ",".join(flagged), pts_k))

        if args.verbose or flagged:
            mstr = " ".join(f"{m.name.split('_',1)[0]}={m.predicted}"
                            for m in models)
            tag = "MISMATCH" if flagged else "ok"
            print(f"  [{tag}] clip {it.clip_idx:3d} N={it.intended_frame:6d} "
                  f"value={row.value:14.4f} dur={int(row.duration):5d} "
                  f"{mstr} pts_k={pts_k} delta={pts_delta:+d}")

    if not otio_mismatches:
        ok("OTIO: every emitted clip frame-perfect under all models "
           "+ PTS bracket")
    else:
        fail(f"OTIO: {len(otio_mismatches)} clips have at least one "
             f"reader model or PTS bracket disagreeing")

    # --- Summary --------------------------------------------------------
    header("Summary")
    overall_pass = (not fcpxml_mismatches) and (not otio_mismatches)
    if overall_pass:
        ok("RESULT: PASS — all emitted clips land on intended frame under "
           "every plausible reader model and the source's actual PTS table.")
        ok("If Resolve still misaligns, the cause is outside the exporter "
           "(Resolve project settings, source-file issue we couldn't see).")
        return 0
    else:
        if fcpxml_mismatches:
            print(f"  FCPXML mismatched clips: "
                  f"{len(fcpxml_mismatches)}/{len(intended)}")
            for idx, n, models, pk in fcpxml_mismatches[:10]:
                print(f"    clip {idx} (intended N={n}): {models} "
                      f"[pts_bracket={pk}]")
            if len(fcpxml_mismatches) > 10:
                print(f"    ... and {len(fcpxml_mismatches) - 10} more")
        if otio_mismatches:
            print(f"  OTIO   mismatched clips: "
                  f"{len(otio_mismatches)}/{len(intended)}")
            for idx, n, models, pk in otio_mismatches[:10]:
                print(f"    clip {idx} (intended N={n}): {models} "
                      f"[pts_bracket={pk}]")
            if len(otio_mismatches) > 10:
                print(f"    ... and {len(otio_mismatches) - 10} more")
        fail("RESULT: FAIL — see flagged columns above to narrow the fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
