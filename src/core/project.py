import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from core.clip import Clip
from core.video_source import VideoSource

logger = logging.getLogger(__name__)

PROJECT_VERSION = 1


@dataclass
class ProjectData:
    sources: List[VideoSource] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)
    playhead_position: int = 0
    selection_follows_playhead: bool = True
    version: int = PROJECT_VERSION


def save_project(filepath: str, sources: dict, clips: list, playhead: int = 0,
                 selection_follows: bool = True):
    data = {
        "version": PROJECT_VERSION,
        "playhead_position": playhead,
        "selection_follows_playhead": selection_follows,
        "sources": [],
        "clips": [],
    }
    for s in sources.values():
        data["sources"].append({
            "id": s.id,
            "file_path": s.file_path,
            "total_frames": s.total_frames,
            "fps": s.fps,
            "width": s.width,
            "height": s.height,
            "codec": s.codec,
        })
    for c in clips:
        data["clips"].append(c.to_dict())

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Project saved to %s", filepath)


def load_project(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = {}
    for sd in data.get("sources", []):
        s = VideoSource(
            id=sd["id"],
            file_path=sd["file_path"],
            total_frames=sd["total_frames"],
            fps=sd["fps"],
            width=sd["width"],
            height=sd["height"],
            codec=sd["codec"],
        )
        sources[s.id] = s

    clips = []
    for cd in data.get("clips", []):
        clips.append(Clip.from_dict(cd))

    return {
        "sources": sources,
        "clips": clips,
        "playhead_position": data.get("playhead_position", 0),
        "selection_follows_playhead": data.get("selection_follows_playhead", True),
        "version": data.get("version", 1),
    }
