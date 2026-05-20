"""Keyframe primitives used to animate ``CropRegion`` geometry over time.

A ``KeyframeTrack`` is an ordered list of ``Keyframe`` records sampled in
fractional source-frame space. Tracks are stored per-axis on a
``CropRegion`` (one each for x, y, w, h) so the animated value at any
source frame can be reconstructed without re-running the editor.

Time domain: keys are pinned to **source frames** of the source video.
Moving a crop's ``anchor_frame`` does not move its keys — the export
window slides over the same animated track. Empty tracks return ``None``
from ``sample()`` so callers fall back to the region's static base
value, which keeps legacy projects (no tracks at all) rendering exactly
as they did before keyframes existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


INTERP_LINEAR = "linear"
INTERP_BEZIER = "bezier"
INTERP_STEP = "step"
_VALID_INTERPS = (INTERP_LINEAR, INTERP_BEZIER, INTERP_STEP)


@dataclass
class Keyframe:
    """One animated value at one source frame.

    ``in_handle`` / ``out_handle`` are (dx, dy) offsets used only when
    ``interp == "bezier"``. ``dx`` is in source-frame units (positive
    extends forward in time), ``dy`` is in value units (same scale as
    ``value``).
    """

    source_frame: int
    value: float
    interp: str = INTERP_LINEAR
    in_handle: Tuple[float, float] = (0.0, 0.0)
    out_handle: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "source_frame": int(self.source_frame),
            "value": float(self.value),
            "interp": self.interp if self.interp in _VALID_INTERPS
            else INTERP_LINEAR,
            "in_handle": [float(self.in_handle[0]),
                          float(self.in_handle[1])],
            "out_handle": [float(self.out_handle[0]),
                           float(self.out_handle[1])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        ih = d.get("in_handle") or [0.0, 0.0]
        oh = d.get("out_handle") or [0.0, 0.0]
        interp = str(d.get("interp", INTERP_LINEAR))
        if interp not in _VALID_INTERPS:
            interp = INTERP_LINEAR
        return cls(
            source_frame=int(d.get("source_frame", 0)),
            value=float(d.get("value", 0.0)),
            interp=interp,
            in_handle=(float(ih[0]), float(ih[1])),
            out_handle=(float(oh[0]), float(oh[1])),
        )


@dataclass
class KeyframeTrack:
    """Ordered list of ``Keyframe`` (sorted by ``source_frame``)."""

    keys: List[Keyframe] = field(default_factory=list)

    # --- introspection ------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def has_key_at(self, source_frame: int) -> bool:
        for k in self.keys:
            if k.source_frame == source_frame:
                return True
        return False

    def find_key(self, source_frame: int) -> Optional[Keyframe]:
        for k in self.keys:
            if k.source_frame == source_frame:
                return k
        return None

    def prev_key_frame(self, source_frame: int) -> Optional[int]:
        best = None
        for k in self.keys:
            if k.source_frame < source_frame:
                best = k.source_frame
            else:
                break
        return best

    def next_key_frame(self, source_frame: int) -> Optional[int]:
        for k in self.keys:
            if k.source_frame > source_frame:
                return k.source_frame
        return None

    # --- mutation -----------------------------------------------------

    def set_key(self, source_frame: int, value: float,
                interp: str = INTERP_LINEAR) -> Keyframe:
        """Insert or replace a key at ``source_frame``. Existing handles
        are preserved when an interp is unchanged; switching interp
        zeroes the handles."""
        existing = self.find_key(source_frame)
        if existing is not None:
            if existing.interp != interp:
                existing.in_handle = (0.0, 0.0)
                existing.out_handle = (0.0, 0.0)
            existing.value = float(value)
            existing.interp = interp
            return existing
        k = Keyframe(source_frame=int(source_frame),
                     value=float(value), interp=interp)
        # Maintain sorted order without importing bisect (no key= until 3.10
        # everywhere we ship).
        i = 0
        while i < len(self.keys) and self.keys[i].source_frame < k.source_frame:
            i += 1
        self.keys.insert(i, k)
        return k

    def remove_key(self, source_frame: int) -> bool:
        for i, k in enumerate(self.keys):
            if k.source_frame == source_frame:
                del self.keys[i]
                return True
        return False

    def toggle_key(self, source_frame: int, value: float,
                   interp: str = INTERP_LINEAR) -> str:
        """Add a key at ``source_frame`` if none exists, remove if one
        does. Returns ``"added"`` or ``"removed"``."""
        if self.has_key_at(source_frame):
            self.remove_key(source_frame)
            return "removed"
        self.set_key(source_frame, value, interp)
        return "added"

    def move_key(self, old_frame: int, new_frame: int,
                 new_value: float) -> bool:
        """Move a key from ``old_frame`` to ``new_frame``, updating its
        value. If a key already exists at ``new_frame`` it is replaced.
        Handles and interp follow the moved key."""
        existing = self.find_key(old_frame)
        if existing is None:
            return False
        ih, oh, interp = existing.in_handle, existing.out_handle, existing.interp
        self.remove_key(old_frame)
        if self.has_key_at(new_frame):
            self.remove_key(new_frame)
        k = self.set_key(new_frame, new_value, interp)
        k.in_handle = ih
        k.out_handle = oh
        return True

    def set_handles(self, source_frame: int,
                    in_handle: Tuple[float, float],
                    out_handle: Tuple[float, float]) -> bool:
        k = self.find_key(source_frame)
        if k is None:
            return False
        k.in_handle = (float(in_handle[0]), float(in_handle[1]))
        k.out_handle = (float(out_handle[0]), float(out_handle[1]))
        return True

    # --- evaluation ---------------------------------------------------

    def sample(self, source_frame: float) -> Optional[float]:
        """Interpolated value at a (possibly fractional) source frame.

        Returns ``None`` when the track is empty so callers can fall
        back to the region's static base value. Outside the keyed range
        the track holds the first / last key value (clamp, not extrap).
        """
        if not self.keys:
            return None
        if len(self.keys) == 1:
            return self.keys[0].value
        if source_frame <= self.keys[0].source_frame:
            return self.keys[0].value
        if source_frame >= self.keys[-1].source_frame:
            return self.keys[-1].value
        # Find bracketing pair.
        for i in range(len(self.keys) - 1):
            a = self.keys[i]
            b = self.keys[i + 1]
            if a.source_frame <= source_frame <= b.source_frame:
                return _interpolate_pair(a, b, source_frame)
        return self.keys[-1].value

    # --- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {"keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "KeyframeTrack":
        if not d:
            return cls()
        raw = d.get("keys") or []
        keys = [Keyframe.from_dict(item) for item in raw]
        keys.sort(key=lambda k: k.source_frame)
        return cls(keys=keys)


# --- Interpolation -------------------------------------------------------

def _interpolate_pair(a: Keyframe, b: Keyframe, t: float) -> float:
    """Sample between bracketing keyframes ``a`` (left) and ``b`` (right)
    at source frame ``t``. The interp mode of the LEFT key drives the
    segment — that's the convention: "this key's outgoing curve."
    """
    span = b.source_frame - a.source_frame
    if span <= 0:
        return a.value
    u = (t - a.source_frame) / span  # 0 .. 1

    if a.interp == INTERP_STEP:
        return a.value
    # A segment is bezier if EITHER endpoint is bezier: the left key's
    # out_handle and the right key's in_handle each shape their own side.
    # (Driving this off the left key alone meant a key's in_handle did
    # nothing whenever its previous neighbour was linear.) A non-bezier
    # endpoint has zero handles, so its control point sits on the key —
    # no pull from that side.
    if a.interp == INTERP_BEZIER or b.interp == INTERP_BEZIER:
        return _bezier_sample(a, b, t)
    # Linear default.
    return a.value + (b.value - a.value) * u


def _bezier_sample(a: Keyframe, b: Keyframe, t: float) -> float:
    """Cubic Bezier on (frame, value). Control points are derived from
    ``a.out_handle`` (offset from ``a``) and ``b.in_handle`` (offset
    from ``b``). The curve passes through ``a`` and ``b`` at the
    segment endpoints.
    """
    p0_x = float(a.source_frame)
    p0_y = float(a.value)
    p3_x = float(b.source_frame)
    p3_y = float(b.value)
    p1_x = p0_x + float(a.out_handle[0])
    p1_y = p0_y + float(a.out_handle[1])
    p2_x = p3_x + float(b.in_handle[0])
    p2_y = p3_y + float(b.in_handle[1])

    # Find parametric u in [0,1] such that x(u) == t. Newton-Raphson with
    # binary-search fallback. The x curve is monotonic when handles are
    # well-formed; we clamp aggressively for the pathological case.
    u = _bezier_invert_x(p0_x, p1_x, p2_x, p3_x, float(t))
    omu = 1.0 - u
    return (omu * omu * omu * p0_y
            + 3.0 * omu * omu * u * p1_y
            + 3.0 * omu * u * u * p2_y
            + u * u * u * p3_y)


def _bezier_invert_x(p0: float, p1: float, p2: float, p3: float,
                     target: float, iters: int = 8) -> float:
    """Return u in [0,1] with x(u) ~= target. Newton-Raphson; falls
    back to bisection if the derivative collapses."""
    if target <= p0:
        return 0.0
    if target >= p3:
        return 1.0
    u = (target - p0) / max(1e-9, p3 - p0)  # linear seed
    for _ in range(iters):
        omu = 1.0 - u
        x = (omu * omu * omu * p0
             + 3.0 * omu * omu * u * p1
             + 3.0 * omu * u * u * p2
             + u * u * u * p3)
        dx = (3.0 * omu * omu * (p1 - p0)
              + 6.0 * omu * u * (p2 - p1)
              + 3.0 * u * u * (p3 - p2))
        if abs(dx) < 1e-6:
            break
        u -= (x - target) / dx
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
    return u
