import logging
import subprocess
import threading
from typing import List, Optional

import av
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.clip import Clip
from core.video_source import VideoSource
from core.proxy_cache import ProxyFile

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
FALLBACK_HSV_THRESHOLD = 30.0

# TransNetV2 constants
TNET_WIDTH = 48
TNET_HEIGHT = 27
TNET_WINDOW = 100
TNET_STEP = 50
TNET_PAD = 25

def _build_ffmpeg_cmd(file_path: str, width: int, height: int) -> List[str]:
    """Build ffmpeg decode command with GPU-accelerated decode + CPU scale.
    Uses -hwaccel cuda for NVDEC decode (fast), vf scale for exact output size."""
    return [
        "ffmpeg", "-v", "quiet",
        "-hwaccel", "cuda",
        "-i", file_path,
        "-vf", f"scale={width}:{height}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def _build_ffmpeg_cmd_cpu(file_path: str, width: int, height: int) -> List[str]:
    """CPU-only fallback."""
    return [
        "ffmpeg", "-v", "quiet", "-threads", "0",
        "-i", file_path,
        "-vf", f"scale={width}:{height}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


class SceneDetector(QThread):
    """Detects shot boundaries using TransNetV2 (GPU-accelerated neural network).
    Uses NVDEC GPU decoding when available for fast frame extraction.
    Falls back to HSV frame differencing if TransNetV2 is unavailable."""

    progress = Signal(int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, source: VideoSource, threshold: float = DEFAULT_THRESHOLD,
                 parent=None):
        super().__init__(parent)
        self._source = source
        self._threshold = threshold
        self._cancelled = False
        self._proc = None  # ffmpeg subprocess reference for cleanup

    def cancel(self):
        self._cancelled = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self):
        try:
            cuts = self._detect_with_transnet()
            if cuts is None:
                logger.info("TransNetV2 unavailable, falling back to HSV method")
                cuts = self._detect_with_hsv()
            if self._cancelled:
                return
            clips = self._cuts_to_clips(cuts)
            self.finished.emit(clips)
        except Exception as e:
            logger.exception("Scene detection failed")
            self.error.emit(str(e))

    # --- TransNetV2 (streaming pipeline with GPU decode) ---

    def _detect_with_transnet(self) -> Optional[List[int]]:
        """Streaming TransNetV2: GPU-decode with ffmpeg in a background thread,
        run GPU inference on windows as frames arrive."""
        try:
            import torch
            from transnetv2_pytorch import TransNetV2
        except ImportError:
            return None

        try:
            model = TransNetV2(device="auto")
            logger.info("TransNetV2 loaded on %s", model.device)
        except Exception as e:
            logger.warning("TransNetV2 failed to initialize: %s", e)
            return None

        total = self._source.total_frames
        frame_size = TNET_WIDTH * TNET_HEIGHT * 3

        # Try GPU-accelerated decode, fall back to CPU
        cmd = _build_ffmpeg_cmd(self._source.file_path, TNET_WIDTH, TNET_HEIGHT)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._proc = proc

        # Quick check: if GPU decode fails, retry with CPU
        test_data = proc.stdout.read(frame_size * 2)
        if len(test_data) < frame_size and proc.poll() is not None:
            logger.info("GPU decode unavailable, using CPU")
            cmd = _build_ffmpeg_cmd_cpu(self._source.file_path, TNET_WIDTH, TNET_HEIGHT)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._proc = proc
            test_data = b""

        # Read all frames in a background thread
        all_frames = []
        read_done = threading.Event()

        def reader_thread():
            buf = bytearray(test_data)  # include any test data already read
            chunk_bytes = frame_size * 500
            try:
                while True:
                    raw = proc.stdout.read(chunk_bytes)
                    if not raw:
                        break
                    buf.extend(raw)
                    while len(buf) >= frame_size:
                        all_frames.append(
                            np.frombuffer(bytes(buf[:frame_size]), dtype=np.uint8)
                            .reshape(TNET_HEIGHT, TNET_WIDTH, 3)
                        )
                        del buf[:frame_size]
                        # Report decode progress
                        n = len(all_frames)
                        if n % max(1, total // 50) == 0:
                            self.progress.emit(min(55, int(n / total * 55)))
            finally:
                proc.stdout.close()
                read_done.set()

        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()

        # Wait for decode to finish
        t.join()
        proc.wait()

        if self._cancelled:
            return []

        n_frames = len(all_frames)
        if n_frames == 0:
            logger.warning("No frames decoded")
            return None

        logger.info("Decoded %d frames, running inference", n_frames)
        self.progress.emit(55)

        # Build padded tensor
        first_frame = all_frames[0]
        last_frame = all_frames[-1]
        pad_end = TNET_PAD + TNET_STEP - (n_frames % TNET_STEP if n_frames % TNET_STEP != 0 else TNET_STEP)

        padded = np.empty((TNET_PAD + n_frames + pad_end, TNET_HEIGHT, TNET_WIDTH, 3), dtype=np.uint8)
        padded[:TNET_PAD] = first_frame
        for i, f in enumerate(all_frames):
            padded[TNET_PAD + i] = f
        padded[TNET_PAD + n_frames:] = last_frame

        # Save all decoded frames as a scrub proxy file (mmap for instant access)
        ProxyFile.save_frames(self._source, all_frames)
        all_frames.clear()

        self.progress.emit(60)

        padded_tensor = torch.from_numpy(np.array(padded, copy=True)).to(model.device)
        del padded

        # Run inference in windows
        predictions = []
        ptr = 0
        total_windows = max(1, (len(padded_tensor) - TNET_WINDOW) // TNET_STEP + 1)

        with torch.no_grad():
            window_idx = 0
            while ptr + TNET_WINDOW <= len(padded_tensor):
                if self._cancelled:
                    return []
                batch = padded_tensor[ptr:ptr + TNET_WINDOW].unsqueeze(0)
                single_pred, _ = model.predict_raw(batch)
                predictions.append(single_pred[0, TNET_PAD:TNET_PAD + TNET_STEP, 0].cpu())
                ptr += TNET_STEP
                window_idx += 1

                if window_idx % max(1, total_windows // 30) == 0:
                    pct = 60 + int(window_idx / total_windows * 38)
                    self.progress.emit(min(98, pct))

        all_preds = torch.cat(predictions, 0)[:n_frames].numpy()
        cut_frames = np.where(all_preds > self._threshold)[0].tolist()

        self.progress.emit(100)
        logger.info("TransNetV2 detected %d cuts in %s (threshold=%.2f)",
                     len(cut_frames), self._source.file_path, self._threshold)
        return cut_frames

    # --- HSV fallback ---

    def _detect_with_hsv(self) -> List[int]:
        """Fallback: detect cuts using HSV frame differencing."""
        container = av.open(self._source.file_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        cuts: List[int] = []
        prev_hsv = None
        frame_index = 0
        total = self._source.total_frames
        hsv_threshold = FALLBACK_HSV_THRESHOLD

        for packet in container.demux(stream):
            if self._cancelled:
                break
            for frame in packet.decode():
                if self._cancelled:
                    break
                rgb = frame.to_ndarray(format="rgb24")
                small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
                if prev_hsv is not None:
                    diff = np.mean(np.abs(hsv - prev_hsv))
                    if diff > hsv_threshold:
                        cuts.append(frame_index)
                prev_hsv = hsv
                frame_index += 1
                if total > 0 and frame_index % max(1, total // 100) == 0:
                    self.progress.emit(min(100, int(frame_index / total * 100)))

        container.close()
        self.progress.emit(100)
        logger.info("HSV fallback detected %d cuts in %s", len(cuts), self._source.file_path)
        return cuts

    # --- Common ---

    def _cuts_to_clips(self, cuts: List[int]) -> List[Clip]:
        total = self._source.total_frames
        if total <= 0:
            return []
        # TransNetV2 reports the last frame of each shot as the cut point.
        # Shift by +1 so boundaries fall on the first frame of the new shot.
        shifted = [c + 1 for c in cuts if c + 1 < total]
        boundaries = [0] + sorted(set(shifted)) + [total]
        clips = []
        for i in range(len(boundaries) - 1):
            frame_in = boundaries[i]
            frame_out = boundaries[i + 1] - 1
            if frame_out < frame_in:
                continue
            clips.append(Clip(
                source_id=self._source.id,
                source_in=frame_in,
                source_out=frame_out,
            ))
        return clips
