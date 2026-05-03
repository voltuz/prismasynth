"""Per-source representative thumbnail extraction for the Media Panel.

Extracts a single frame from the middle of each imported video and caches it
to ``%LOCALAPPDATA%/prismasynth/cache/source_thumbs/<source-id>.jpg``. Cache
is shared across sessions and projects.

Decoupled from ``ThumbnailCache`` (which generates per-clip thumbnails for the
timeline strip): this only produces ONE image per source and is invoked at
import time, not on a sweep.
"""

import logging
import os
import subprocess
from pathlib import Path

from core.video_source import VideoSource

logger = logging.getLogger(__name__)

THUMB_WIDTH = 160
THUMB_HEIGHT = 90


def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home())
    d = Path(base) / "prismasynth" / "cache" / "source_thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path_for(source: VideoSource) -> Path:
    """Where the thumbnail for the given source lives on disk."""
    return _cache_dir() / f"{source.id}.jpg"


def extract_thumbnail(source: VideoSource, *, force: bool = False) -> Path:
    """Extract a single mid-clip thumbnail and cache it. Returns the path
    (whether freshly extracted or already cached). Returns the path even on
    failure — caller is responsible for handling a missing/empty file.

    force=True re-extracts even if a cached thumbnail already exists.
    """
    out = cache_path_for(source)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out

    fps = source.fps if source.fps > 0 else 24.0
    mid_seconds = (source.total_frames / fps) / 2.0 if source.total_frames > 0 else 0.0

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{mid_seconds:.3f}",
        "-i", source.file_path,
        "-frames:v", "1",
        "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={THUMB_WIDTH}:{THUMB_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
        "-q:v", "4",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0:
            logger.warning("source thumb extract failed for %s: %s",
                           source.file_path, proc.stderr.decode("utf-8", "replace")[:200])
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("source thumb extract error for %s: %s", source.file_path, e)
    return out
