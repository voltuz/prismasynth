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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0
