import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Clip:
    source_id: Optional[str]
    source_in: int   # first frame (inclusive)
    source_out: int  # last frame (inclusive)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = ""
    color_index: int = 0
    # Group (People) memberships. Stored in toggle order; the timeline
    # label strip sorts by digit at paint time for stable chip ordering.
    group_ids: list = field(default_factory=list)

    @property
    def is_gap(self) -> bool:
        return self.source_id is None

    @property
    def duration_frames(self) -> int:
        return self.source_out - self.source_in + 1

    @classmethod
    def make_gap(cls, duration_frames: int) -> "Clip":
        return cls(
            source_id=None,
            source_in=0,
            source_out=duration_frames - 1,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_in": self.source_in,
            "source_out": self.source_out,
            "label": self.label,
            "color_index": self.color_index,
            "group_ids": list(self.group_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(
            id=d["id"],
            source_id=d.get("source_id"),
            source_in=d["source_in"],
            source_out=d["source_out"],
            label=d.get("label", ""),
            color_index=d.get("color_index", 0),
            group_ids=list(d.get("group_ids", [])),
        )
