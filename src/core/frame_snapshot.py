"""Native-resolution single-frame PNG snapshot.

Wraps a one-shot ffmpeg invocation that decodes exactly one frame from a
source file at full source resolution and writes a PNG. Reuses the
half-frame seek margin from `core.exporter.Exporter._frame_to_seek_ts`
(see that docstring for the IEEE-754 rationale) so the requested
`frame_num` is the first frame ffmpeg keeps after the accurate seek.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _seek_ts(frame_num: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return max(0.0, (frame_num - 0.5) / fps)


def snapshot_frame_to_png(
    source_path: str,
    frame_num: int,
    fps: float,
    out_path: Path,
) -> Optional[str]:
    """Decode source@frame_num and write it to ``out_path`` as PNG.

    Returns the absolute output path on success, ``None`` on failure.
    Best-effort: ffmpeg errors are logged, never raised.
    """
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("snapshot: cannot create output dir %s: %s", out_path.parent, e)
        return None

    ts = _seek_ts(int(frame_num), float(fps))
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-v", "error",
        "-ss", f"{ts:.6f}",
        "-i", source_path,
        "-frames:v", "1",
        "-update", "1",
        str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("snapshot ffmpeg failed: %s", e)
        return None

    if result.returncode != 0 or not out_path.exists():
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.warning("snapshot ffmpeg rc=%s: %s", result.returncode, stderr)
        return None

    return str(out_path)
