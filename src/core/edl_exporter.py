"""CMX 3600 EDL (Edit Decision List) exporter.

Generates standard EDL files for import into Premiere Pro, DaVinci Resolve,
Avid, and other NLEs.
"""

import os
from typing import Dict

from core.timeline import TimelineModel
from core.video_source import VideoSource


def _frame_to_tc_count(frame: int, fps: float) -> int:
    """Convert frame number to total TC frame count using time-based conversion.
    Uses integer arithmetic to avoid floating-point rounding issues."""
    fps_int = round(fps)
    # For NTSC rates: fps = fps_int * 1000 / 1001
    # frame * 1001 gives ticks at 1/24000s resolution
    # Integer division avoids float rounding
    actual_time = frame / fps
    secs = int(actual_time)
    ff = round((actual_time - secs) * fps_int)
    if ff >= fps_int:
        ff = 0
        secs += 1
    return secs * fps_int + ff


def _tc_count_to_string(tc_total: int, fps_int: int) -> str:
    """Convert TC frame count to HH:MM:SS:FF string."""
    ff = tc_total % fps_int
    total_secs = tc_total // fps_int
    ss = total_secs % 60
    mm = (total_secs // 60) % 60
    hh = total_secs // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def frames_to_timecode(frames: int, fps: float) -> str:
    """Convert frame number to non-drop-frame timecode HH:MM:SS:FF."""
    if fps <= 0:
        fps = 24.0
    return _tc_count_to_string(_frame_to_tc_count(frames, fps), round(fps))


def _reel_name(source: VideoSource) -> str:
    """Generate a reel name from the source file (max 8 chars for CMX 3600)."""
    name = os.path.splitext(os.path.basename(source.file_path))[0]
    # Strip non-alphanumeric, truncate to 8 chars
    clean = "".join(c for c in name if c.isalnum() or c in "-_")
    return (clean[:8] or "AX").ljust(8)


def export_edl(timeline: TimelineModel, sources: Dict[str, VideoSource],
               output_path: str, include_gaps: bool = False,
               use_render_range: bool = False, title: str = "PrismaSynth",
               fps: float = None):
    """Export timeline as a CMX 3600 EDL file.

    Args:
        timeline: The timeline model with clips.
        sources: Dict mapping source_id to VideoSource.
        output_path: Path to write the .edl file.
        include_gaps: If True, gaps advance the record timecode (NLE sees empty space).
                      If False, clips are placed back-to-back (compact timeline).
        use_render_range: If True, only export clips within the in/out render range.
        title: EDL title string.
        fps: Frame rate for timecodes. If None, uses first source's FPS.
    """
    clips = timeline.clips
    if not clips:
        return

    # Determine FPS
    if fps is None:
        first_source = next(iter(sources.values()), None)
        fps = first_source.fps if first_source else 24.0

    # Get render range if applicable
    if use_render_range:
        render_start, render_end = timeline.get_render_range()
    else:
        render_start = 0
        render_end = timeline.get_total_duration_frames() - 1

    lines = []
    lines.append(f"TITLE: {title}")
    lines.append(f"FCM: NON-DROP FRAME")
    lines.append(f"* FRAME RATE: {fps:.3f}")
    lines.append("")

    event_num = 0
    rec_pos = 0  # record (output) position in frames
    rec_tc = 0   # running record TC total (chains perfectly, no gaps)
    timeline_pos = 0  # position in the original timeline

    for clip in clips:
        clip_start = timeline_pos
        clip_end = timeline_pos + clip.duration_frames - 1
        timeline_pos += clip.duration_frames

        # Skip clips outside render range
        if clip_end < render_start or clip_start > render_end:
            continue

        # Compute effective range (clip may be partially in render range)
        eff_start = max(clip_start, render_start)
        eff_end = min(clip_end, render_end)
        eff_duration = eff_end - eff_start + 1

        if clip.is_gap:
            if include_gaps:
                rec_pos += eff_duration
                rec_tc += eff_duration
            continue

        source = sources.get(clip.source_id)
        if source is None:
            continue

        # Source timecodes: SRC_IN is time-based (matches Resolve's PTS),
        # SRC_OUT = SRC_IN_tc + duration (preserves exact frame count)
        offset_in_clip = eff_start - clip_start
        src_in_frame = clip.source_in + offset_in_clip
        fps_int = round(fps)
        src_in_tc = _frame_to_tc_count(src_in_frame, fps)
        src_out_tc = src_in_tc + eff_duration

        # Record timecodes: chained from running total (no gaps)
        rec_in_tc = rec_tc
        rec_out_tc = rec_tc + eff_duration

        event_num += 1
        reel = _reel_name(source)
        filename = os.path.basename(source.file_path)

        lines.append(
            f"{event_num:03d}  {reel}  V     C        "
            f"{_tc_count_to_string(src_in_tc, fps_int)} "
            f"{_tc_count_to_string(src_out_tc, fps_int)} "
            f"{_tc_count_to_string(rec_in_tc, fps_int)} "
            f"{_tc_count_to_string(rec_out_tc, fps_int)}"
        )
        lines.append(f"* FROM CLIP NAME: {filename}")
        lines.append(f"* SOURCE FILE: {source.file_path}")
        lines.append("")

        rec_pos += eff_duration
        rec_tc = rec_out_tc

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
