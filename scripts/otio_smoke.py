"""Smoke test for core.otio_exporter — no UI, no real media.

Builds a minimal in-memory timeline, runs the exporter across the option
matrix (gaps on/off, render-range on/off), parses each result back as
JSON, and asserts structural invariants against the OTIO schema:

  - Top-level ``Timeline.1`` with a ``Stack.1`` containing one video
    ``Track.1``.
  - Clip items use ``Clip.2`` with a ``DEFAULT_MEDIA`` reference.
  - ``source_range`` ``start_time`` carries the +0.25 frame seek nudge
    and ``duration`` stays integer.
  - ``available_range`` on each media reference matches source length
    with no nudge (integer).
  - Gap source_range stays integer (no nudge).
  - ``prismasynth`` metadata round-trips with the right fields.
  - Packed layout (no gaps) drops gap items from the track.
  - Render-range clipping clamps first/last clips to the range.
  - Empty timeline writes nothing.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.clip import Clip
from core.timeline import TimelineModel
from core.video_source import VideoSource
from core.otio_exporter import export_otio

FPS = 24000 / 1001  # 23.976


def _build_model():
    src = VideoSource(
        file_path="C:/videos/movie.mov",
        total_frames=1000,
        fps=FPS,
        width=1920,
        height=1080,
        codec="h264",
    )
    sources = {src.id: src}

    tl = TimelineModel()
    c1 = Clip(source_id=src.id, source_in=100, source_out=199,
              label="intro", color_index=3)     # 100 frames
    gap = Clip.make_gap(50)                      # 50-frame gap
    c2 = Clip(source_id=src.id, source_in=300, source_out=399,
              label="middle", color_index=5)     # 100 frames
    c3 = Clip(source_id=src.id, source_in=500, source_out=549,
              label="tail", color_index=7)       # 50 frames
    tl.add_clips([c1, gap, c2, c3], assign_colors=False)
    return tl, sources, src, (c1, gap, c2, c3)


def _load_track(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["OTIO_SCHEMA"] == "Timeline.1", doc["OTIO_SCHEMA"]
    stack = doc["tracks"]
    assert stack["OTIO_SCHEMA"] == "Stack.1"
    assert len(stack["children"]) == 1
    track = stack["children"][0]
    assert track["OTIO_SCHEMA"] == "Track.1"
    assert track["kind"] == "Video"
    return doc, track


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


NUDGE = 0.25  # must match core.otio_exporter._SRC_SEEK_NUDGE


def _src_range(item):
    """For clips: returns (raw_start_float, integer_duration).
    The raw_start is used to verify the nudge is present, then checked
    against the nominal integer frame via _nominal_frame()."""
    r = item["source_range"]
    return r["start_time"]["value"], int(r["duration"]["value"])


def _nominal_frame(raw_start: float) -> int:
    """Strip the nudge back to the integer frame the writer intended."""
    return int(round(raw_start - NUDGE))


def check_packed():
    tl, sources, src, (c1, gap, c2, c3) = _build_model()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "packed.otio")
        export_otio(tl, sources, out, include_gaps=False)
        doc, track = _load_track(out)

    children = track["children"]
    schemas = [c["OTIO_SCHEMA"] for c in children]
    _assert(schemas == ["Clip.2", "Clip.2", "Clip.2"],
            f"packed schemas: {schemas}")

    # source_range checks — start carries the +NUDGE seek nudge,
    # duration stays integer.
    s, dur = _src_range(children[0])
    _assert(abs(s - (100 + NUDGE)) < 1e-9 and _nominal_frame(s) == 100 and dur == 100,
            f"clip1 range: ({s}, {dur})")
    s, dur = _src_range(children[1])
    _assert(abs(s - (300 + NUDGE)) < 1e-9 and _nominal_frame(s) == 300 and dur == 100,
            f"clip2 range: ({s}, {dur})")
    s, dur = _src_range(children[2])
    _assert(abs(s - (500 + NUDGE)) < 1e-9 and _nominal_frame(s) == 500 and dur == 50,
            f"clip3 range: ({s}, {dur})")

    # available_range on the shared media reference — NOT nudged
    # (it describes the source file, not a seek position).
    ref = children[0]["media_references"]["DEFAULT_MEDIA"]
    _assert(ref["OTIO_SCHEMA"] == "ExternalReference.1",
            f"ref schema: {ref['OTIO_SCHEMA']}")
    _assert(ref["target_url"].startswith("file:///"),
            f"ref target_url: {ref['target_url']}")
    ar = ref["available_range"]
    _assert(ar["duration"]["value"] == 1000.0,
            f"avail dur: {ar['duration']['value']}")
    _assert(ar["start_time"]["value"] == 0.0,
            f"avail start: {ar['start_time']['value']}")

    # active key
    _assert(children[0]["active_media_reference_key"] == "DEFAULT_MEDIA",
            "active key mismatch")

    # metadata round-trip
    md = children[0]["metadata"]["prismasynth"]
    _assert(md["label"] == "intro", f"label: {md['label']}")
    _assert(md["color_index"] == 3, f"color: {md['color_index']}")
    _assert(md["source_id"] == src.id, f"source_id: {md['source_id']}")
    _assert(md["clip_id"] == c1.id, f"clip_id: {md['clip_id']}")

    # Every clip's DEFAULT_MEDIA should be the same dict identity (shared
    # in Python). After JSON round-trip they become equal-but-distinct
    # dicts — structural equality is what we verify here.
    ref2 = children[1]["media_references"]["DEFAULT_MEDIA"]
    _assert(ref == ref2, "media references diverge across clips")
    print("[OK] packed layout")


def check_with_gaps():
    tl, sources, _, _ = _build_model()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "gaps.otio")
        export_otio(tl, sources, out, include_gaps=True)
        _, track = _load_track(out)

    children = track["children"]
    schemas = [c["OTIO_SCHEMA"] for c in children]
    _assert(schemas == ["Clip.2", "Gap.1", "Clip.2", "Clip.2"],
            f"gaps schemas: {schemas}")
    # Gap source_range is NOT nudged — no source seek involved.
    gap_start = children[1]["source_range"]["start_time"]["value"]
    gap_dur = children[1]["source_range"]["duration"]["value"]
    _assert(gap_start == 0.0 and gap_dur == 50.0,
            f"gap range: start={gap_start} dur={gap_dur}")
    print("[OK] gaps included")


def check_render_range():
    """Timeline layout (frames): [c1:0-99][gap:100-149][c2:150-249][c3:250-299].
    in=50, out=249 (inclusive). Packed layout (gaps excluded):
      - c1 clipped to timeline 50..99  -> source_in 150, dur 50
      - gap excluded entirely (packed)
      - c2 kept whole (150..249)      -> source_in 300, dur 100
    """
    tl, sources, _, _ = _build_model()
    tl.set_in_point(50)
    tl.set_out_point(249)

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ranged.otio")
        export_otio(tl, sources, out, include_gaps=False, use_render_range=True)
        _, track = _load_track(out)

    children = track["children"]
    schemas = [c["OTIO_SCHEMA"] for c in children]
    _assert(schemas == ["Clip.2", "Clip.2"], f"ranged schemas: {schemas}")

    s, dur = _src_range(children[0])
    _assert(_nominal_frame(s) == 150 and dur == 50, f"c1 clipped: ({s}, {dur})")

    s, dur = _src_range(children[1])
    _assert(_nominal_frame(s) == 300 and dur == 100, f"c2 kept: ({s}, {dur})")
    print("[OK] render range clip")


def check_empty_guard():
    tl = TimelineModel()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "empty.otio")
        export_otio(tl, {}, out)
        _assert(not os.path.exists(out),
                "empty export should not create file")
    print("[OK] empty timeline guard")


def check_stress_gaps():
    """Multiple non-trivial gaps interspersed with clips at non-aligned
    source ranges. Asserts that the packed-layout offsets sum to the sum
    of non-gap durations and each clip's nudged source_range.start_time
    equals the original clip.source_in. Then re-runs with a render range
    that splits the first and last surviving clips, mirroring the writer
    math from otio_exporter.py:185-205."""
    src = VideoSource(
        file_path="C:/videos/stress.mov",
        total_frames=20000,
        fps=FPS,
        width=1920,
        height=1080,
        codec="h264",
    )
    sources = {src.id: src}

    clips = [
        Clip(source_id=src.id, source_in=50,   source_out=50  + 70  - 1,
             label="c0"),
        Clip.make_gap(13),
        Clip(source_id=src.id, source_in=4444, source_out=4444 + 40 - 1,
             label="c1"),
        Clip.make_gap(7),
        Clip(source_id=src.id, source_in=211,  source_out=211 + 123 - 1,
             label="c2"),
        Clip.make_gap(31),
        Clip(source_id=src.id, source_in=9001, source_out=9001 + 50 - 1,
             label="c3"),
    ]
    real_clips = [c for c in clips if not c.is_gap]
    expected_in = [50, 4444, 211, 9001]
    expected_dur = [70, 40, 123, 50]
    total_real = sum(expected_dur)  # 283

    # --- Packed (no gaps) -----------------------------------------------
    tl = TimelineModel()
    tl.add_clips(clips, assign_colors=False)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "stress_packed.otio")
        export_otio(tl, sources, out, include_gaps=False)
        _, track = _load_track(out)
    children = track["children"]
    schemas = [c["OTIO_SCHEMA"] for c in children]
    _assert(schemas == ["Clip.2"] * 4, f"stress packed schemas: {schemas}")

    running = 0
    for i, child in enumerate(children):
        s, dur = _src_range(child)
        _assert(_nominal_frame(s) == expected_in[i] and dur == expected_dur[i],
                f"stress packed clip {i}: got ({s},{dur}) expected "
                f"(~{expected_in[i] + NUDGE}, {expected_dur[i]})")
        running += dur
    _assert(running == total_real,
            f"stress packed total dur {running} != {total_real}")

    # --- Render range (gaps excluded) -----------------------------------
    # Layout (frames):
    #   c0 0..69  | gap 70..82 | c1 83..122 | gap 123..129
    # | c2 130..252 | gap 253..283 | c3 284..333
    # set in=35, out=220 -> c0 clipped to timeline 35..69 (35 frames),
    # c1 whole (40), c2 clipped to 130..220 (91 frames), c3 dropped.
    tl2 = TimelineModel()
    tl2.add_clips(clips, assign_colors=False)
    tl2.set_in_point(35)
    tl2.set_out_point(220)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "stress_ranged.otio")
        export_otio(tl2, sources, out, include_gaps=False,
                    use_render_range=True)
        _, track = _load_track(out)
    children = track["children"]
    schemas = [c["OTIO_SCHEMA"] for c in children]
    _assert(schemas == ["Clip.2"] * 3, f"stress ranged schemas: {schemas}")

    # c0: timeline 0..69, render in=35 -> offset_in_clip=35, src=50+35=85,
    #     eff_dur = min(69,220)-max(0,35)+1 = 35
    # c1: timeline 83..122, fully inside [35,220] -> src=4444, dur=40
    # c2: timeline 130..252, clipped to 130..220 -> offset_in_clip=0,
    #     src=211, dur=91
    expected = [(85, 35), (4444, 40), (211, 91)]
    for i, child in enumerate(children):
        s, dur = _src_range(child)
        e_in, e_dur = expected[i]
        _assert(_nominal_frame(s) == e_in and dur == e_dur,
                f"stress ranged clip {i}: got ({s},{dur}) expected "
                f"(~{e_in + NUDGE}, {e_dur})")

    print("[OK] stress-gap layout (packed + render-range)")


def main():
    check_packed()
    check_with_gaps()
    check_render_range()
    check_empty_guard()
    check_stress_gaps()
    print("\nAll OTIO smoke checks passed.")


if __name__ == "__main__":
    main()
