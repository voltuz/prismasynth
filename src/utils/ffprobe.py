import json
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoInfo:
    width: int
    height: int
    total_frames: int
    fps: float
    duration_seconds: float
    codec: str


def probe_video(file_path: str) -> Optional[VideoInfo]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        return None

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec = video_stream.get("codec_name", "unknown")

    # Frame count: try nb_frames, then compute from duration * fps
    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        total_frames = int(nb_frames)
    else:
        total_frames = 0  # will be computed later

    # FPS from r_frame_rate (e.g., "24000/1001")
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 24.0  # safe default

    # Guard against zero/negative FPS
    if fps <= 0:
        fps = 24.0

    # Duration: try stream duration, then format duration
    duration = video_stream.get("duration")
    if not duration or duration == "N/A":
        duration = data.get("format", {}).get("duration")
    duration_seconds = float(duration) if duration and duration != "N/A" else 0.0

    # Compute frame count from duration if not available
    if total_frames == 0 and fps > 0 and duration_seconds > 0:
        total_frames = int(duration_seconds * fps)

    return VideoInfo(
        width=width,
        height=height,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        codec=codec,
    )
