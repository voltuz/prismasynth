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
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    # Container time_base of the video stream (e.g. ("1", "16000")). 0/0 if
    # the probe couldn't read it. Used by VideoSource.is_seek_safe to flag
    # sources whose container can't exactly represent their declared fps —
    # FCPXML/OTIO imports of those drift in NLEs that time-seek the source.
    time_base_num: int = 0
    time_base_den: int = 0


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

    tb_num, tb_den = 0, 0
    tb_str = video_stream.get("time_base", "")
    if isinstance(tb_str, str) and "/" in tb_str:
        try:
            n, d = tb_str.split("/", 1)
            tb_num, tb_den = int(n), int(d)
        except ValueError:
            tb_num, tb_den = 0, 0

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

    audio_codec = ""
    audio_sample_rate = 0
    audio_channels = 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_codec = stream.get("codec_name", "") or ""
            try:
                audio_sample_rate = int(stream.get("sample_rate") or 0)
            except (TypeError, ValueError):
                audio_sample_rate = 0
            try:
                audio_channels = int(stream.get("channels") or 0)
            except (TypeError, ValueError):
                audio_channels = 0
            break

    return VideoInfo(
        width=width,
        height=height,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        codec=codec,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        time_base_num=tb_num,
        time_base_den=tb_den,
    )


_hdr_cache: dict = {}


def probe_hdr(file_path: str) -> bool:
    """Check if a video file is HDR (PQ/HLG + BT.2020). Cached per path."""
    if file_path in _hdr_cache:
        return _hdr_cache[file_path]

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            _hdr_cache[file_path] = False
            return False
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        _hdr_cache[file_path] = False
        return False

    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        transfer = stream.get("color_transfer", "")
        primaries = stream.get("color_primaries", "")
        is_hdr = (transfer in ("smpte2084", "arib-std-b67")
                  and primaries == "bt2020")
        _hdr_cache[file_path] = is_hdr
        return is_hdr

    _hdr_cache[file_path] = False
    return False
