import uuid
from dataclasses import dataclass, field
from typing import Tuple


def _frame_duration_for_fps(fps: float) -> Tuple[int, int]:
    """Rational ``(num, den)`` for one frame's duration in seconds at ``fps``.

    Mirrors ``core.xml_exporter._rate_to_frame_duration`` for the standard
    NTSC + integer rates we care about. Kept local to avoid importing the
    exporter module from ``video_source`` (would create a cycle, since
    ``xml_exporter`` already imports ``video_source``).
    """
    if fps <= 0:
        return (100, 2400)
    if abs(fps - 24000 / 1001) < 0.01:
        return (1001, 24000)
    if abs(fps - 30000 / 1001) < 0.01:
        return (1001, 30000)
    if abs(fps - 60000 / 1001) < 0.01:
        return (1001, 60000)
    if abs(fps - round(fps)) < 0.001:
        n = int(round(fps))
        return (100, n * 100)
    return (1000, int(round(fps * 1000)))


def is_seek_safe(time_base_num: int, time_base_den: int, fps: float) -> bool:
    """True iff a container with ``time_base = num/den`` can represent every
    frame's PTS exactly at ``fps``. Returns True (safe) when the time_base is
    unknown so absent metadata never trips the warning.

    For NTSC 23.976 (24000/1001), `1/16000` averages 667.333 ticks per frame
    and is unsafe; `1/24000` and `1/24000000` are exact and safe. Resolve
    seeks the source by *time*, so an unsafe source-side tick grid means our
    FCPXML/OTIO ``start`` rationals round to the wrong source frame.
    """
    if time_base_num <= 0 or time_base_den <= 0:
        return True
    fnum, fden = _frame_duration_for_fps(fps)
    return (fnum * time_base_den) % (fden * time_base_num) == 0


@dataclass
class VideoSource:
    file_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    codec: str
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    # Video-stream container time_base (e.g. 1/16000). 0/0 = unknown
    # (legacy projects, probe failure). See is_seek_safe.
    time_base_num: int = 0
    time_base_den: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0

    @property
    def has_audio(self) -> bool:
        return self.audio_channels > 0

    @property
    def time_base_str(self) -> str:
        """Human-readable timebase, or empty string when unknown."""
        if self.time_base_num <= 0 or self.time_base_den <= 0:
            return ""
        return f"{self.time_base_num}/{self.time_base_den}"

    def is_seek_safe(self) -> bool:
        return is_seek_safe(self.time_base_num, self.time_base_den, self.fps)

    def format_audio(self) -> str:
        """Short human-readable audio description, or 'none' if the source
        has no audio. Used by the clip info panel and the export dialogs."""
        if not self.has_audio:
            return "none"
        parts = []
        if self.audio_codec:
            parts.append(self.audio_codec)
        parts.append(f"{self.audio_channels} ch")
        if self.audio_sample_rate > 0:
            khz = self.audio_sample_rate / 1000.0
            parts.append(f"{khz:g} kHz")
        return ", ".join(parts)
