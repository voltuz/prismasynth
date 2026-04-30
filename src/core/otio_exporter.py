"""OpenTimelineIO (OTIO) exporter — native .otio JSON.

Emits a Timeline document in OTIO's own public schema so it can be consumed
directly by DaVinci Resolve Studio (first-class OTIO import) and any tool
that uses the OTIO library. For Premiere Pro, route the .otio through an
OTIO adapter (AAF / Premiere XML) after export — that conversion lives
outside of PrismaSynth.

We write the JSON by hand rather than importing ``opentimelineio``:
  - OTIO's Python package ships a native (pybind11) extension whose wheel
    tagging is broken on Python 3.14 — the build produces ``cp313`` .pyd
    files that the 3.14 interpreter refuses to load. Avoiding the
    dependency dodges the issue and keeps requirements.txt light.
  - The on-disk schema is stable and well-specified (Timeline.1, Stack.1,
    Track.1, Clip.2, Gap.1, ExternalReference.1, TimeRange.1,
    RationalTime.1). All major OTIO releases since 0.15 consume the output
    we emit.

Alongside the existing FCPXML exporter:
  - FCPXML is tuned for Resolve's NTSC seek rounding (see
    ``xml_exporter._src_seek_str``) and stays our recommended Resolve path.
  - OTIO gives a clean, NLE-agnostic interchange for everything else, with
    exact integer frame counts at the sequence rate (no time-vs-frame
    conversion at import).

Per-clip metadata:
  ``metadata["prismasynth"]`` round-trips ``source_id``, ``clip_id``,
  ``color_index``, and ``label`` so a future OTIO importer could
  reconstruct a .psynth project with no loss.

NTSC / fractional-rate seek nudge:
  ``source_range.start_time.value`` for each clip is written as
  ``frame + 0.25``, not ``frame``. *Why:* when an OTIO reader converts
  our (value, rate) to seconds and back to a frame, floating-point
  round-trip at rates like 24000/1001 produces errors of up to ~1 ULP —
  enough to land ``N.0`` at ``N-1.9999999...``. Observed empirically
  against DaVinci Resolve Studio: out of 16 clips in test_09.otio,
  clip 5 (source start 3543) drifted to 3542 on import. Sweeping frames
  0-10000 with Resolve's rate (23.976023976023979 vs our
  23.976023976023978) shows 692 frames drift by -1; nudging by +0.25
  eliminates all of them and stays safely clear of any integer boundary
  under ``floor()``, ``trunc()``, and ``round()`` reader behaviour.
  +0.5 would match FCPXML's ``_src_seek_str`` but breaks readers that
  use ``round()`` — fractional residues just over 0.5 snap to frame N+1.
  +0.25 sits squarely in frame N's interior for every rounding mode.
  ``duration``, ``available_range``, and gaps are NOT nudged — they are
  counts / file descriptors, not source-seek positions.
"""

import json
import os
from pathlib import Path
from typing import Dict

from core.timeline import TimelineModel
from core.video_source import VideoSource

# See module docstring: value that source_range.start_time is offset by so
# that a reader's (value/rate)*reader_rate round-trip never lands just
# below frame N.
_SRC_SEEK_NUDGE = 0.25


def _file_uri(path: str) -> str:
    return Path(os.path.abspath(path)).as_uri()


def _rational_time(value: float, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "RationalTime.1",
        "rate": rate,
        "value": float(value),
    }


def _time_range(start: float, duration: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": _rational_time(duration, rate),
        "start_time": _rational_time(start, rate),
    }


def _external_reference(url: str, avail_frames: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "ExternalReference.1",
        "metadata": {},
        "name": "",
        "available_range": _time_range(0, avail_frames, rate),
        "available_image_bounds": None,
        "target_url": url,
    }


def _clip_element(name: str, source_start: int, duration: int, rate: float,
                  media_ref: dict, ps_meta: dict) -> dict:
    # Clip.2 uses a media_references dict + active key. Shipped in OTIO
    # 0.15 (2021); every current adapter reads it.
    return {
        "OTIO_SCHEMA": "Clip.2",
        "metadata": {"prismasynth": ps_meta},
        "name": name,
        "source_range": _time_range(source_start, duration, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
        "media_references": {"DEFAULT_MEDIA": media_ref},
        "active_media_reference_key": "DEFAULT_MEDIA",
    }


def _gap_element(duration: int, rate: float) -> dict:
    return {
        "OTIO_SCHEMA": "Gap.1",
        "metadata": {},
        "name": "",
        "source_range": _time_range(0, duration, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
    }


def export_otio(timeline: TimelineModel, sources: Dict[str, VideoSource],
                output_path: str, include_gaps: bool = False,
                use_render_range: bool = False, title: str = "PrismaSynth",
                fps: float = None):
    """Export timeline as an OpenTimelineIO (.otio) JSON file.

    Args:
        timeline: Timeline model with clips.
        sources: ``source_id -> VideoSource`` for every imported source.
        output_path: Path to write the .otio file.
        include_gaps: If True, gaps become ``Gap`` items on the track.
                      If False, clips are packed back-to-back.
        use_render_range: If True, only clips inside the in/out range are
                          exported.
        title: Timeline name stored in the OTIO file.
        fps: Sequence frame rate. If None, uses first source's FPS.
    """
    clips = timeline.clips
    if not clips:
        return

    if fps is None:
        first_source = next(iter(sources.values()), None)
        fps = first_source.fps if first_source else 24.0

    if use_render_range:
        render_start, render_end = timeline.get_render_range()
    else:
        render_start = 0
        render_end = timeline.get_total_duration_frames() - 1

    # One ExternalReference dict per source; every clip that shares a
    # source points at the same dict so the JSON stays compact.
    media_refs: Dict[str, dict] = {}

    def _media_ref_for(source_id: str) -> dict:
        ref = media_refs.get(source_id)
        if ref is not None:
            return ref
        src = sources[source_id]
        ref = _external_reference(
            url=_file_uri(src.file_path),
            avail_frames=src.total_frames,
            rate=fps,
        )
        ref["name"] = os.path.splitext(os.path.basename(src.file_path))[0]
        media_refs[source_id] = ref
        return ref

    track_children: list = []
    timeline_pos = 0
    emitted_any = False

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
                track_children.append(_gap_element(eff_duration, fps))
                emitted_any = True
            continue

        source = sources.get(clip.source_id)
        if source is None:
            continue

        offset_in_clip = eff_start - clip_start
        src_in_frame = clip.source_in + offset_in_clip
        clip_name = os.path.splitext(os.path.basename(source.file_path))[0]

        track_children.append(_clip_element(
            name=clip_name,
            source_start=src_in_frame + _SRC_SEEK_NUDGE,
            duration=eff_duration,
            rate=fps,
            media_ref=_media_ref_for(clip.source_id),
            ps_meta={
                "source_id": clip.source_id,
                "clip_id": clip.id,
                "color_index": clip.color_index,
                "label": clip.label,
            },
        ))
        emitted_any = True

    if not emitted_any:
        return

    document = {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {},
        "name": title,
        "global_start_time": None,
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "metadata": {},
            "name": "tracks",
            "source_range": None,
            "effects": [],
            "markers": [],
            "enabled": True,
            "children": [
                {
                    "OTIO_SCHEMA": "Track.1",
                    "metadata": {},
                    "name": "Video",
                    "source_range": None,
                    "effects": [],
                    "markers": [],
                    "enabled": True,
                    "children": track_children,
                    "kind": "Video",
                }
            ],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=4)
