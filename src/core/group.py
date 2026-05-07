"""People / group tagging.

A `Group` is a project-scoped tag that clips can belong to (zero or more
groups per clip). Each group has a display name, a colour shown on the
timeline label strip, and an optional digit (0-9) that maps to the
keyboard shortcuts ``group_digit_0``-``group_digit_9`` for fast tagging.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


# Palette used by TimelineModel.add_group when no explicit colour is given.
# Distinct enough from the timeline-clip palette that group labels read as
# a separate visual layer.
GROUP_COLOR_PALETTE = [
    "#5577aa", "#aa5577", "#77aa55", "#aa7755",
    "#55aa77", "#7755aa", "#aaaa55", "#55aaaa",
    "#aa5555", "#5555aa", "#aa55aa", "#55aa55",
]


@dataclass
class Group:
    name: str
    color: str           # hex string, e.g. "#5577aa"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    digit: Optional[int] = None  # 0-9, unique across groups; None = no key

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "digit": self.digit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Group":
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            color=d.get("color", "#888888"),
            digit=d.get("digit"),
        )


def clip_matches_filter(clip, group_filter) -> bool:
    """Decide whether a clip should be included under a group filter.

    Filter encoding:
      - ``None``                                → no filter (all clips pass)
      - ``{"group_ids": [...], "include_untagged": bool}``
                                                → filter active

    Match rule:
      - No filter active → always match.
      - Clip with no groups → match iff ``include_untagged`` is true.
      - Clip with groups → match iff at least one of its groups is in
        ``group_ids``.
    """
    if group_filter is None:
        return True
    gids = clip.group_ids
    if not gids:
        return bool(group_filter.get("include_untagged", False))
    selected = set(group_filter.get("group_ids", []))
    return any(g in selected for g in gids)
