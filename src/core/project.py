import json
import logging
import os
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
                 selection_follows: bool = True,
                 in_point: Optional[int] = None, out_point: Optional[int] = None,
                 scroll_offset: int = 0):
    data = {
        "version": PROJECT_VERSION,
        "playhead_position": playhead,
        "scroll_offset": scroll_offset,
        "selection_follows_playhead": selection_follows,
        "in_point": in_point,
        "out_point": out_point,
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

    # Write to a temp file then atomically rename, so a crash or power loss
    # mid-write (especially under the 60s autosave) cannot destroy the
    # existing project file.
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)
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
        "scroll_offset": data.get("scroll_offset", 0),
        "selection_follows_playhead": data.get("selection_follows_playhead", True),
        "in_point": data.get("in_point"),
        "out_point": data.get("out_point"),
        "version": data.get("version", 1),
    }
