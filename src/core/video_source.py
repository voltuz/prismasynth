import uuid
from dataclasses import dataclass, field


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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0

    @property
    def has_audio(self) -> bool:
        return self.audio_channels > 0

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
