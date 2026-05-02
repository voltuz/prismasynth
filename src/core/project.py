import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from core.clip import Clip
from core.video_source import VideoSource
from utils.ffprobe import probe_video

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
    project_dir = os.path.dirname(filepath)
    for s in sources.values():
        try:
            relative_path = os.path.relpath(s.file_path, start=project_dir)
        except ValueError:
            # Cross-drive on Windows raises ValueError — no relative path possible.
            relative_path = None
        data["sources"].append({
            "id": s.id,
            "file_path": s.file_path,
            "relative_path": relative_path,
            "total_frames": s.total_frames,
            "fps": s.fps,
            "width": s.width,
            "height": s.height,
            "codec": s.codec,
            "audio_codec": s.audio_codec,
            "audio_sample_rate": s.audio_sample_rate,
            "audio_channels": s.audio_channels,
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

    project_dir = os.path.dirname(filepath)
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
            audio_codec=sd.get("audio_codec", ""),
            audio_sample_rate=sd.get("audio_sample_rate", 0),
            audio_channels=sd.get("audio_channels", 0),
        )
        # If the absolute path is gone, try the relative path (project + sources
        # transferred together). Only the relink dialog handles the rest.
        if not os.path.exists(s.file_path):
            rel = sd.get("relative_path")
            if rel:
                candidate = os.path.normpath(os.path.join(project_dir, rel))
                if os.path.exists(candidate):
                    s.file_path = candidate
        # Retroactive audio probe: legacy projects (saved before audio support)
        # have no audio fields. Re-probe so the project becomes audio-aware on
        # first load — no manual migration step.
        if not s.audio_codec and s.audio_channels == 0:
            try:
                info = probe_video(s.file_path)
            except Exception:
                info = None
            if info is not None:
                s.audio_codec = info.audio_codec
                s.audio_sample_rate = info.audio_sample_rate
                s.audio_channels = info.audio_channels
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
