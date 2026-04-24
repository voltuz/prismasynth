"""Final Cut Pro XML (FCPXML) exporter.

FCPXML 1.9 — supported natively by DaVinci Resolve, Premiere Pro (via Adobe
Dynamic Link), Final Cut Pro. Unlike CMX 3600 EDL, FCPXML uses rational time
fractions (e.g. ``1001/24000s`` for a single 23.976 frame) so frame positions
are expressed exactly. There is no timecode-to-frame conversion on the
importer side — Resolve reads the fraction directly — so clips land on the
exact source frame we intended, with no NDF/drift ambiguity.
"""

import os
from pathlib import Path
from typing import Dict, Tuple
from xml.sax.saxutils import escape as xml_escape

from core.timeline import TimelineModel
from core.video_source import VideoSource


def _rate_to_frame_duration(fps: float) -> Tuple[int, int]:
    """Return (numerator, denominator) of a single frame's duration in seconds.

    Emit the TRUE frame duration for the declared fps:
      - 23.976 (24000/1001) → 1001/24000 s
      - 29.97  (30000/1001) → 1001/30000 s
      - 59.94  (60000/1001) → 1001/60000 s
      - 24                 → 100/2400 s
      - 25                 → 1/25 s
      - 30                 → 100/3000 s
      - 50                 → 1/50 s
      - 60                 → 100/6000 s

    We tried using the integer-fps base (1/24s) for NTSC 23.976 to dodge
    Resolve's 23.976/24 conflation. That was WRONG for well-formed files:
    when Resolve seeks the source by time, writing 2769/24s (115.375s)
    makes Resolve load file frame round(115.375 × 23.976) = 2766 — a
    -3-frame drift. The accurate 2769*1001/24000s (115.490s) maps through
    time-based seek to frame 2769 exactly. Verified empirically.

    Remaining ±1 drift in multi-clip Resolve imports traces to the
    SOURCE FILE's container timebase, not our XML: MOVs exported before
    PrismaSynth v0.2.0 used time_base=1/16000 which can't represent
    23.976 exactly (667 vs 668 ticks per frame averaging), accumulating
    rounding artifacts. v0.2.0+ exports use time_base=1/24000000 which
    is exact; those files round-trip through Resolve cleanly.
    """
    if fps <= 0:
        return (100, 2400)
    if abs(fps - 24000 / 1001) < 0.01:
        return (1001, 24000)
    if abs(fps - 30000 / 1001) < 0.01:
        return (1001, 30000)
    if abs(fps - 60000 / 1001) < 0.01:
        return (1001, 60000)
    # Integer rates: use 100/N*100 so numerator > 1 (Resolve prefers this).
    if abs(fps - round(fps)) < 0.001:
        n = int(round(fps))
        return (100, n * 100)
    return (1000, int(round(fps * 1000)))


def _time_str(frames: int, frame_num: int, frame_den: int) -> str:
    """Format a frame count as an FCPXML rational time string.

    ``frames * (num/den) s`` = ``(frames*num)/den s``. We keep the denominator
    fixed at ``frame_den`` so every time in the document shares a common time
    base, which Resolve handles cleanly without normalization surprises.
    """
    if frames == 0:
        return "0s"
    return f"{frames * frame_num}/{frame_den}s"


def _src_seek_str(frame: int, frame_num: int, frame_den: int) -> str:
    """Format an asset-clip ``start`` (source seek position) with an NTSC nudge.

    *Why the nudge:* Resolve imports NTSC 23.976 assets with an internal
    ``frameDuration="1/24s"`` (ignoring our declared ``1001/24000s``). When it
    stores our asset-clip ``start`` it rounds our ``N*1001/24000`` fraction to
    the nearest ``1/24`` tick. For ``N mod 1000 < 500`` that rounds DOWN past
    frame N's PTS, and the subsequent time-based seek loads file frame N-1
    — exactly the "first frame from the previous clip" drift observed
    against real Resolve imports of test_06.psynth / test_08.psynth.

    Shifting our numerator by ``+frame_num/2`` lands the start value near the
    middle of frame N's PTS range. That's well inside the range for both
    rounding directions: if Resolve preserves the fraction the seek still
    hits frame N (offset half a frame into its range); if Resolve rounds to
    the nearest ``1/24`` tick, the ``+0.5`` fractional shift forces round-up,
    which seeks into frame N from above. Verified drift-free on all 16
    clips of test_08.psynth.

    For pure-integer frame rates (24, 25, 30, 50, 60) the nudge is a no-op
    for Resolve's rounding (no 23.976/24 mismatch exists), but lands the
    seek mid-frame anyway — harmless.
    """
    if frame <= 0:
        return "0s"
    num = frame * frame_num + frame_num // 2
    return f"{num}/{frame_den}s"


def _file_uri(path: str) -> str:
    """Convert a native file path to a file:// URI suitable for FCPXML ``src``."""
    return Path(os.path.abspath(path)).as_uri()


def _asset_id(prefix: str, n: int) -> str:
    return f"{prefix}{n}"


def _tc_format(fps: float) -> str:
    """Resolve uses tcFormat to pick timecode display; underlying frame math is
    independent of this. NTSC rates -> NDF for simplicity (we don't write
    drop-frame timecodes ourselves; display preference only)."""
    # DF is only valid for 29.97 / 59.94. For 23.976 must be NDF. Keep NDF
    # everywhere — it's the safe choice and matches how our EDL path worked.
    return "NDF"


def export_fcpxml(timeline: TimelineModel, sources: Dict[str, VideoSource],
                  output_path: str, include_gaps: bool = False,
                  use_render_range: bool = False, title: str = "PrismaSynth",
                  fps: float = None):
    """Export timeline as an FCPXML 1.9 file.

    Args:
        timeline: The timeline model with clips.
        sources: Dict mapping source_id to VideoSource.
        output_path: Path to write the .fcpxml file.
        include_gaps: If True, gaps appear as ``<gap>`` elements on the spine.
                      If False, clips are placed back-to-back (compact).
        use_render_range: If True, only export clips within the in/out range.
        title: Project/event name.
        fps: Sequence frame rate. If None, uses first source's FPS.
    """
    clips = timeline.clips
    if not clips:
        return

    # Determine FPS
    if fps is None:
        first_source = next(iter(sources.values()), None)
        fps = first_source.fps if first_source else 24.0

    # Render range
    if use_render_range:
        render_start, render_end = timeline.get_render_range()
    else:
        render_start = 0
        render_end = timeline.get_total_duration_frames() - 1

    frame_num, frame_den = _rate_to_frame_duration(fps)
    tc_format = _tc_format(fps)

    # Assign stable asset IDs. One asset per source used in the export.
    # Resource IDs: r1 = format, r2+ = assets.
    format_id = "r1"
    asset_ids: Dict[str, str] = {}

    def _get_asset_id(source_id: str) -> str:
        if source_id not in asset_ids:
            asset_ids[source_id] = _asset_id("r", 2 + len(asset_ids))
        return asset_ids[source_id]

    # First pass: determine which sources are referenced so we only emit
    # assets for them (an unused asset in FCPXML is valid but noisy).
    used_source_ids = []
    seen = set()
    timeline_pos = 0
    for clip in clips:
        clip_start = timeline_pos
        clip_end = timeline_pos + clip.duration_frames - 1
        timeline_pos += clip.duration_frames
        if clip_end < render_start or clip_start > render_end:
            continue
        if clip.is_gap:
            continue
        if clip.source_id not in seen and clip.source_id in sources:
            seen.add(clip.source_id)
            used_source_ids.append(clip.source_id)

    if not used_source_ids:
        return

    # Allocate asset IDs in discovery order.
    for sid in used_source_ids:
        _get_asset_id(sid)

    # --- Build XML ---

    ref_fps = frame_num / frame_den  # sanity
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE fcpxml>')
    lines.append('<fcpxml version="1.9">')

    # Resources
    lines.append('  <resources>')
    fmt_name = f"FFVideoFormat{_fmt_name_suffix(fps)}"
    # Pull a representative width/height from the first used asset.
    ref_source = sources[used_source_ids[0]]
    lines.append(
        f'    <format id="{format_id}" name="{xml_escape(fmt_name)}" '
        f'frameDuration="{frame_num}/{frame_den}s" '
        f'width="{ref_source.width}" height="{ref_source.height}" '
        f'colorSpace="1-1-1 (Rec. 709)"/>'
    )
    for sid in used_source_ids:
        src = sources[sid]
        aid = asset_ids[sid]
        asset_duration = _time_str(src.total_frames, frame_num, frame_den)
        name = os.path.splitext(os.path.basename(src.file_path))[0]
        uri = _file_uri(src.file_path)
        # Resolve's preferred asset-source syntax is a <media-rep> child element
        # rather than an `src=` attribute. Matches what Resolve writes on export
        # and is more likely to be respected round-trip.
        # Path.as_uri() already percent-encodes reserved chars, so the URI
        # won't contain &, <, >, or " — safe to embed directly.
        lines.append(
            f'    <asset id="{aid}" name="{xml_escape(name)}" '
            f'start="0s" duration="{asset_duration}" '
            f'hasVideo="1" format="{format_id}" '
            f'videoSources="1">'
        )
        lines.append(
            f'      <media-rep kind="original-media" src="{uri}"/>'
        )
        lines.append('    </asset>')
    lines.append('  </resources>')

    # Compute sequence duration — total output length.
    # If gaps included: clips land at their in-timeline positions (within
    # render range). If not: packed back-to-back.
    if include_gaps:
        seq_duration_frames = render_end - render_start + 1
    else:
        seq_duration_frames = 0
        tp = 0
        for clip in clips:
            clip_start = tp
            clip_end = tp + clip.duration_frames - 1
            tp += clip.duration_frames
            if clip_end < render_start or clip_start > render_end:
                continue
            if clip.is_gap:
                continue
            eff_start = max(clip_start, render_start)
            eff_end = min(clip_end, render_end)
            seq_duration_frames += eff_end - eff_start + 1

    seq_duration = _time_str(seq_duration_frames, frame_num, frame_den)

    # Library / event / project / sequence
    lines.append('  <library>')
    lines.append(f'    <event name="{xml_escape(title)}">')
    lines.append(f'      <project name="{xml_escape(title)} Timeline">')
    lines.append(
        f'        <sequence format="{format_id}" duration="{seq_duration}" '
        f'tcStart="0s" tcFormat="{tc_format}" '
        f'audioLayout="stereo" audioRate="48k">'
    )
    lines.append('          <spine>')

    # Spine: one asset-clip per real clip. Offset = position on timeline.
    timeline_pos = 0
    rec_pos_frames = 0  # used when gaps are excluded (packed layout)
    gap_counter = 0
    clip_counter = 0

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
            if include_gaps:
                gap_offset = eff_start - render_start
                lines.append(
                    f'            <gap name="Gap" '
                    f'offset="{_time_str(gap_offset, frame_num, frame_den)}" '
                    f'start="0s" '
                    f'duration="{_time_str(eff_duration, frame_num, frame_den)}"/>'
                )
                gap_counter += 1
            continue

        source = sources.get(clip.source_id)
        if source is None:
            continue

        offset_in_clip = eff_start - clip_start
        src_in_frame = clip.source_in + offset_in_clip

        if include_gaps:
            spine_offset_frames = eff_start - render_start
        else:
            spine_offset_frames = rec_pos_frames

        aid = asset_ids[clip.source_id]
        clip_name = os.path.splitext(os.path.basename(source.file_path))[0]
        clip_counter += 1
        lines.append(
            f'            <asset-clip ref="{aid}" '
            f'name="{xml_escape(clip_name)}" '
            f'offset="{_time_str(spine_offset_frames, frame_num, frame_den)}" '
            f'start="{_src_seek_str(src_in_frame, frame_num, frame_den)}" '
            f'duration="{_time_str(eff_duration, frame_num, frame_den)}" '
            f'format="{format_id}" '
            f'tcFormat="{tc_format}"/>'
        )

        rec_pos_frames += eff_duration

    lines.append('          </spine>')
    lines.append('        </sequence>')
    lines.append('      </project>')
    lines.append('    </event>')
    lines.append('  </library>')
    lines.append('</fcpxml>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt_name_suffix(fps: float) -> str:
    """Produce a human-readable format name suffix Resolve expects
    (e.g. ``1080p2398``). Purely cosmetic — the format element's frameDuration
    is what Resolve actually uses for timing."""
    if abs(fps - 24000 / 1001) < 0.01:
        return "2398"
    if abs(fps - 24) < 0.01:
        return "24"
    if abs(fps - 25) < 0.01:
        return "25"
    if abs(fps - 30000 / 1001) < 0.01:
        return "2997"
    if abs(fps - 30) < 0.01:
        return "30"
    if abs(fps - 50) < 0.01:
        return "50"
    if abs(fps - 60000 / 1001) < 0.01:
        return "5994"
    if abs(fps - 60) < 0.01:
        return "60"
    return str(int(round(fps)))
