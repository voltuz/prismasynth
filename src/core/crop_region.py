"""Per-clip rectangular crop regions for the cropping-export feature.

Each ``CropRegion`` is one rectangle drawn on a single clip in source-pixel
space. At export time it produces an 81-frame, 16fps video (frames dropped
from the source's native FPS — no slow-down) cropped to the rectangle.

Coordinate space is **native source pixels** so crops stay zoom/pan-stable
in the preview and resolution-stable across project moves.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple

from core.keyframe import KeyframeTrack


# --- Fixed output shape (training pipeline requirement) --------------------

OUTPUT_FRAMES = 81
OUTPUT_FPS = 16
AUDIO_SAMPLE_RATE = 48000


def required_source_frames(src_fps: float) -> int:
    """Minimum source frames a clip needs to host one 81-frame crop at 16fps.

    ``ceil(81 * src_fps / 16)`` — must be at least this many frames between
    ``anchor_frame`` and the clip's ``source_out`` (inclusive) for ffmpeg's
    ``fps=16`` filter to emit a full 81-frame output.
    """
    if src_fps <= 0:
        return OUTPUT_FRAMES
    return int(math.ceil(OUTPUT_FRAMES * src_fps / OUTPUT_FPS))


def exact_audio_samples_81_at_16(sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
    """Exact sample count for an 81-frame@16fps window.

    ``81 * sample_rate // 16`` — integer, no NTSC rational math needed
    because the output rate is exactly 16fps regardless of source fps.
    For 48 kHz this is exactly 243000.
    """
    return OUTPUT_FRAMES * sample_rate // OUTPUT_FPS


# --- Aspect ratio --------------------------------------------------------

# Built-in aspect-ratio presets (label -> (num, den)). "free" = no lock,
# "custom" = use CropRegion.custom_ratio_w / custom_ratio_h.
ASPECT_PRESETS = {
    "free":   None,
    "1:1":    (1, 1),
    "4:3":    (4, 3),
    "16:9":   (16, 9),
    "9:16":   (9, 16),
    "3:4":    (3, 4),
    "custom": None,  # uses custom_ratio_w/h
}


def resolve_aspect(cr: "CropRegion") -> Optional[tuple]:
    """Return (num, den) for the crop's aspect-ratio lock, or None if free."""
    if cr.aspect_ratio == "custom":
        if cr.custom_ratio_w > 0 and cr.custom_ratio_h > 0:
            return (cr.custom_ratio_w, cr.custom_ratio_h)
        return None
    return ASPECT_PRESETS.get(cr.aspect_ratio)


# --- Export segment ------------------------------------------------------

@dataclass
class Segment:
    """One 81-frame@16fps export window of a crop region.

    A crop region's keyframe animation is shared across all its
    segments; each segment just picks which source-frame slice
    (``[anchor_frame, anchor_frame + required_source_frames)``) gets
    exported. ``active=False`` segments render dimmed and are skipped on
    export."""

    anchor_frame: int = 0
    active: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "anchor_frame": int(self.anchor_frame),
            "active": bool(self.active),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:12],
            anchor_frame=int(d.get("anchor_frame", 0)),
            active=bool(d.get("active", True)),
        )


# --- Data class ----------------------------------------------------------

@dataclass
class CropRegion:
    # Source-pixel rectangle.
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    # See ASPECT_PRESETS. "custom" uses custom_ratio_w/h.
    aspect_ratio: str = "free"
    custom_ratio_w: int = 0
    custom_ratio_h: int = 0

    # Optional single-group tag. None = "Untagged" bucket.
    group_id: Optional[str] = None

    # Inactive crops render dashed in the preview and are skipped on export.
    active: bool = True

    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Export windows. A crop always has at least one. Each segment is an
    # 81-frame@16fps window anchored at a source frame; the crop's
    # animation tracks are shared across all of them.
    segments: list = field(default_factory=lambda: [Segment()])

    # Per-axis keyframe tracks. Empty track → axis falls back to the
    # static base value above. Time domain is source frames of the
    # source video (NOT clip-local, NOT timeline-local), so the keys
    # stay pinned to the same source content when ``anchor_frame``
    # moves or the clip's in/out shifts.
    x_track: KeyframeTrack = field(default_factory=KeyframeTrack)
    y_track: KeyframeTrack = field(default_factory=KeyframeTrack)
    w_track: KeyframeTrack = field(default_factory=KeyframeTrack)
    h_track: KeyframeTrack = field(default_factory=KeyframeTrack)

    # --- Animated geometry sampling --------------------------------

    def track_for(self, axis: str) -> KeyframeTrack:
        """Map axis name → track. Raises KeyError on unknown axis."""
        return {"x": self.x_track, "y": self.y_track,
                "w": self.w_track, "h": self.h_track}[axis]

    def base_value(self, axis: str) -> int:
        return {"x": self.x, "y": self.y,
                "w": self.w, "h": self.h}[axis]

    def find_segment(self, segment_id: str) -> Optional["Segment"]:
        for s in self.segments:
            if s.id == segment_id:
                return s
        return None

    def is_animated(self) -> bool:
        """True iff any axis track has at least one keyframe."""
        return bool(self.x_track or self.y_track
                    or self.w_track or self.h_track)

    def sample(self, source_frame: float) -> Tuple[int, int, int, int]:
        """Return interpolated (x, y, w, h) at the given source frame.

        Empty tracks fall back to the region's static base value. The
        result is always int-cast so callers (overlay rendering, exporter
        per-frame geometry) get pixel-aligned output.

        When an aspect-ratio lock is active, the lock is **authoritative**:
        height is derived from width so the sampled rectangle always has
        the locked ratio, regardless of any drift between the w/h keyframe
        tracks (e.g. tracks edited while the crop was "free", then locked,
        or the W/H curves edited independently in the graph editor). This
        is non-destructive — the stored tracks are untouched, so switching
        the lock back to "free" restores the original animation. Because
        every consumer (preview, export, segment_aspect_constant,
        crop_output_dims) goes through sample(), they all stay consistent.
        """
        sx = self.x_track.sample(source_frame)
        sy = self.y_track.sample(source_frame)
        sw = self.w_track.sample(source_frame)
        sh = self.h_track.sample(source_frame)
        x = self.x if sx is None else sx
        y = self.y if sy is None else sy
        w = self.w if sw is None else sw
        h = self.h if sh is None else sh
        ar = resolve_aspect(self)
        if ar is not None and ar[0] > 0 and ar[1] > 0:
            h = w * ar[1] / ar[0]  # width drives height; lock wins
        return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    # --- Serialization ---------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "segments": [s.to_dict() for s in self.segments],
            "aspect_ratio": self.aspect_ratio,
            "custom_ratio_w": int(self.custom_ratio_w),
            "custom_ratio_h": int(self.custom_ratio_h),
            "group_id": self.group_id,
            "active": bool(self.active),
            "label": self.label,
        }
        # Only persist tracks that actually contain keys so legacy
        # projects round-trip byte-identical.
        if self.x_track:
            d["x_track"] = self.x_track.to_dict()
        if self.y_track:
            d["y_track"] = self.y_track.to_dict()
        if self.w_track:
            d["w_track"] = self.w_track.to_dict()
        if self.h_track:
            d["h_track"] = self.h_track.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CropRegion":
        # Segments: prefer the new list; migrate legacy single
        # ``anchor_frame``; fall back to one default segment. Always
        # guarantee at least one segment.
        raw_segments = d.get("segments")
        if raw_segments:
            segments = [Segment.from_dict(s) for s in raw_segments]
        elif "anchor_frame" in d:
            segments = [Segment(anchor_frame=int(d.get("anchor_frame", 0)))]
        else:
            segments = [Segment()]
        if not segments:
            segments = [Segment()]
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:12],
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 0)),
            h=int(d.get("h", 0)),
            segments=segments,
            aspect_ratio=str(d.get("aspect_ratio", "free")),
            custom_ratio_w=int(d.get("custom_ratio_w", 0)),
            custom_ratio_h=int(d.get("custom_ratio_h", 0)),
            group_id=d.get("group_id") or None,
            active=bool(d.get("active", True)),
            label=str(d.get("label", "")),
            x_track=KeyframeTrack.from_dict(d.get("x_track")),
            y_track=KeyframeTrack.from_dict(d.get("y_track")),
            w_track=KeyframeTrack.from_dict(d.get("w_track")),
            h_track=KeyframeTrack.from_dict(d.get("h_track")),
        )


# --- Validation helpers --------------------------------------------------

def clip_source_frames(clip) -> int:
    """Total source frames the clip spans (inclusive)."""
    return clip.source_out - clip.source_in + 1


def can_host_crop(clip, src_fps: float) -> bool:
    """True iff the clip is long enough to host a single crop at this fps."""
    if clip is None or clip.is_gap:
        return False
    return clip_source_frames(clip) >= required_source_frames(src_fps)


def clamp_anchor(anchor: int, clip, src_fps: float) -> int:
    """Clamp ``anchor`` to the valid range for a crop on ``clip``.

    Valid range is ``[clip.source_in, clip.source_out - required + 1]``.
    Returns the input unchanged if the clip is too short to host any crop
    (callers should pre-check with ``can_host_crop``).
    """
    if clip is None or clip.is_gap:
        return anchor
    req = required_source_frames(src_fps)
    lo = clip.source_in
    hi = clip.source_out - req + 1
    if hi < lo:
        return anchor
    if anchor < lo:
        return lo
    if anchor > hi:
        return hi
    return anchor


def crop_matches_filter(crop: "CropRegion", group_filter) -> bool:
    """Decide whether a crop should be included under a group filter.

    Mirrors ``core.group.clip_matches_filter`` but treats the crop's single
    optional ``group_id`` as its membership set. Encoding:
      - ``None``                                → no filter (all crops pass)
      - ``{"group_ids": [...], "include_untagged": bool}``
                                                → filter active
    """
    if group_filter is None:
        return True
    if crop.group_id is None:
        return bool(group_filter.get("include_untagged", False))
    selected = set(group_filter.get("group_ids", []))
    return crop.group_id in selected


# --- Animated-export geometry --------------------------------------------

# Max relative drift in the sampled w/h ratio across an export window for the
# segment to still count as "constant aspect ratio" (and therefore
# exportable). Genuine aspect keyframes change the ratio by tens of percent;
# int-rounding of the per-frame w/h only wobbles it by a fraction of a
# percent, so a 3% gate cleanly separates the two.
ASPECT_DRIFT_TOL = 0.03


def _to_even(value: float) -> int:
    """Round to the nearest positive even integer (encoder chroma-stride
    requirement for yuv420). Never returns below 2."""
    return max(2, int(round(value / 2.0)) * 2)


def segment_aspect_constant(cr: "CropRegion", anchor_frame: int,
                            src_fps: float,
                            tol: float = ASPECT_DRIFT_TOL) -> bool:
    """True iff the crop's width/height ratio stays within ``tol`` across the
    81-frame@16fps window anchored at ``anchor_frame``.

    Non-animated crops are trivially constant. Animated crops that change
    *size* with a fixed aspect ratio (pan, or aspect-locked zoom) pass;
    crops that change *shape* (aspect ratio) mid-window fail and must be
    skipped on export — the fixed-resolution output can't represent a
    changing aspect ratio without distortion.
    """
    if not cr.is_animated():
        return True
    n_src = required_source_frames(src_fps) + 2
    ratios = []
    for k in range(n_src):
        _, _, w, h = cr.sample(anchor_frame + k)
        if w <= 0 or h <= 0:
            return False
        ratios.append(w / h)
    lo, hi = min(ratios), max(ratios)
    if lo <= 0:
        return False
    return (hi - lo) / lo <= tol


def crop_output_dims(cr: "CropRegion", anchor_frame: int,
                     out_width: int) -> Tuple[int, int]:
    """Even ``(out_w, out_h)`` for a crop exported at ``out_width`` pixels
    wide, preserving the crop's aspect ratio (sampled at the anchor — which
    is constant for any exportable crop). Height follows the ratio so crops
    of differing shapes never get distorted by a single global width."""
    _, _, w, h = cr.sample(anchor_frame)
    if w <= 0 or h <= 0:
        w, h = cr.w or 1, cr.h or 1
    out_w = _to_even(out_width)
    out_h = _to_even(out_width * h / w)
    return out_w, out_h


def crop_native_dims(cr: "CropRegion", anchor_frame: int,
                     src_fps: float) -> Tuple[int, int]:
    """Even ``(w, h)`` = the LARGEST crop size across the segment's window
    (the most zoomed-out / biggest-box frame).

    For 'native' output mode: sizing each file to its biggest frame means
    that frame is ~1:1 and nothing is ever downscaled (smaller frames scale
    up to match). Aspect ratio is constant for any exportable crop, so the
    max width and its paired height occur on the same frame."""
    n_src = required_source_frames(src_fps) + 2
    best_w, best_h = 0, 0
    for k in range(n_src):
        _, _, w, h = cr.sample(int(anchor_frame) + k)
        if w > best_w:
            best_w, best_h = w, h
    if best_w <= 0:
        best_w, best_h = cr.w or 2, cr.h or 2
    return _to_even(best_w), _to_even(best_h)
