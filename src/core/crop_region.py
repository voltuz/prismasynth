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
from typing import Optional


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


# --- Data class ----------------------------------------------------------

@dataclass
class CropRegion:
    # Source-pixel rectangle.
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    # First source frame of the 81-output-frame window.
    anchor_frame: int = 0

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "anchor_frame": int(self.anchor_frame),
            "aspect_ratio": self.aspect_ratio,
            "custom_ratio_w": int(self.custom_ratio_w),
            "custom_ratio_h": int(self.custom_ratio_h),
            "group_id": self.group_id,
            "active": bool(self.active),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CropRegion":
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:12],
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 0)),
            h=int(d.get("h", 0)),
            anchor_frame=int(d.get("anchor_frame", 0)),
            aspect_ratio=str(d.get("aspect_ratio", "free")),
            custom_ratio_w=int(d.get("custom_ratio_w", 0)),
            custom_ratio_h=int(d.get("custom_ratio_h", 0)),
            group_id=d.get("group_id") or None,
            active=bool(d.get("active", True)),
            label=str(d.get("label", "")),
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
