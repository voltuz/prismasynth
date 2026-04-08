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

# Parallel decode
N_DECODE_SEGMENTS = 4


def _build_ffmpeg_cmd(file_path: str, width: int, height: int,
                      ss: float = None, duration: float = None) -> List[str]:
    """Build ffmpeg decode command with GPU-accelerated decode + CPU scale."""
    cmd = ["ffmpeg", "-v", "quiet", "-hwaccel", "cuda"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.4f}"]
    cmd += ["-i", file_path]
    if duration is not None:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += ["-vf", f"scale={width}:{height}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


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
    Uses parallel NVDEC decoding for speed. Falls back to HSV if TransNetV2 unavailable."""

    progress = Signal(int)            # percent (0-100)
    detail_progress = Signal(int, int, str)  # frames_done, total_frames, phase
    phase_changed = Signal(str)       # emitted when switching stages
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, source: VideoSource, threshold: float = DEFAULT_THRESHOLD,
                 frame_range: tuple = None, parent=None):
        super().__init__(parent)
        self._source = source
        self._threshold = threshold
        # Optional (start_frame, end_frame) to limit detection range
        self._frame_range = frame_range
        self._cancelled = False
        self._procs: list = []

    def cancel(self):
        self._cancelled = True
        # Kill all ffmpeg subprocesses immediately
        procs = list(self._procs)
        for proc in procs:
            try:
                proc.kill()
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

    # --- TransNetV2 ---

    def _detect_with_transnet(self) -> Optional[List[int]]:
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

        if self._frame_range:
            range_start, range_end = self._frame_range
            total = range_end - range_start + 1
        else:
            range_start = 0
            total = self._source.total_frames
        pad_end = TNET_PAD + TNET_STEP - (total % TNET_STEP if total % TNET_STEP != 0 else TNET_STEP)

        # Pre-allocate padded array — all decode paths write directly into it
        padded = np.empty((TNET_PAD + total + pad_end, TNET_HEIGHT, TNET_WIDTH, 3), dtype=np.uint8)

        # Decode frames — try parallel ffmpeg, then single
        n_frames = self._decode_parallel(padded, total, range_start)
        if n_frames == 0:
            n_frames = self._decode_single(padded, total, range_start)

        if self._cancelled or n_frames == 0:
            return [] if self._cancelled else None

        # Pad edges
        padded[:TNET_PAD] = padded[TNET_PAD]
        padded[TNET_PAD + n_frames:] = padded[TNET_PAD + n_frames - 1]

        if self._cancelled:
            return []

        # Save legacy proxy for scrub fallback
        frames_list = [padded[TNET_PAD + i] for i in range(n_frames)]
        ProxyFile.save_frames(self._source, frames_list)
        del frames_list

        if self._cancelled:
            return []

        logger.info("Decoded %d frames, running inference", n_frames)
        self.progress.emit(100)
        self.phase_changed.emit("Analyzing")
        self.progress.emit(0)

        # Transfer to GPU and run inference
        padded_tensor = torch.from_numpy(padded).to(model.device)
        del padded

        if self._cancelled:
            del padded_tensor
            return []

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

                if window_idx % max(1, total_windows // 100) == 0:
                    pct = int(window_idx / total_windows * 100)
                    self.progress.emit(min(99, pct))
                    frames_done = min(n_frames, window_idx * TNET_STEP)
                    self.detail_progress.emit(frames_done, n_frames, "Analyzing")

        all_preds = torch.cat(predictions, 0)[:n_frames].numpy()
        cut_frames = np.where(all_preds > self._threshold)[0].tolist()

        self.progress.emit(100)
        logger.info("TransNetV2 detected %d cuts in %s (threshold=%.2f)",
                     len(cut_frames), self._source.file_path, self._threshold)
        return cut_frames

    # --- Decode: from JPEG proxy (fastest, ~628 fps) ---

    # --- Decode: parallel ffmpeg segments (~396 fps) ---

    def _decode_parallel(self, padded: np.ndarray, total: int,
                         range_start: int = 0) -> int:
        """Decode using N parallel ffmpeg/NVDEC processes."""
        fps = self._source.fps if self._source.fps > 0 else 24.0
        frame_size = TNET_WIDTH * TNET_HEIGHT * 3
        n_seg = N_DECODE_SEGMENTS
        frames_per_seg = total // n_seg

        seg_counts = [0] * n_seg
        seg_errors = [None] * n_seg
        progress_lock = threading.Lock()
        total_decoded = [0]

        def decode_segment(seg_idx, start_frame, max_frames):
            ss = start_frame / fps
            dur = (max_frames + 100) / fps
            cmd = _build_ffmpeg_cmd(self._source.file_path, TNET_WIDTH, TNET_HEIGHT,
                                    ss=ss, duration=dur)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._procs.append(proc)

            buf = bytearray()
            count = 0
            offset = TNET_PAD + start_frame
            try:
                while count < max_frames and not self._cancelled:
                    raw = proc.stdout.read(frame_size * 500)
                    if not raw:
                        break
                    buf.extend(raw)
                    while len(buf) >= frame_size and count < max_frames:
                        padded[offset + count] = (
                            np.frombuffer(bytes(buf[:frame_size]), np.uint8)
                            .reshape(TNET_HEIGHT, TNET_WIDTH, 3)
                        )
                        del buf[:frame_size]
                        count += 1

                        with progress_lock:
                            total_decoded[0] += 1
                            n = total_decoded[0]
                        if n % max(1, total // 200) == 0:
                            self.progress.emit(min(100, int(n / total * 100)))
                            self.detail_progress.emit(n, total, "Decoding")
            except Exception as e:
                seg_errors[seg_idx] = e
            finally:
                try:
                    proc.stdout.close()
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            seg_counts[seg_idx] = count

        # Test if GPU decode works with a quick probe
        test_cmd = _build_ffmpeg_cmd(self._source.file_path, TNET_WIDTH, TNET_HEIGHT)
        test_proc = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        test_data = test_proc.stdout.read(frame_size * 2)
        test_proc.stdout.close()
        test_proc.kill()
        test_proc.wait()
        if len(test_data) < frame_size:
            logger.info("GPU decode unavailable for parallel, falling back to single")
            return 0

        threads = []
        for i in range(n_seg):
            start = range_start + i * frames_per_seg
            count = frames_per_seg if i < n_seg - 1 else total - (i * frames_per_seg)
            t = threading.Thread(target=decode_segment, args=(i, start, count))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        n_frames = sum(seg_counts)
        errors = [e for e in seg_errors if e is not None]
        if errors:
            logger.warning("Segment decode errors: %s", errors)

        return n_frames

    # --- Decode: single ffmpeg (fallback, ~262 fps) ---

    def _decode_single(self, padded: np.ndarray, total: int,
                       range_start: int = 0) -> int:
        """Single ffmpeg process decode — CPU fallback."""
        frame_size = TNET_WIDTH * TNET_HEIGHT * 3
        fps = self._source.fps if self._source.fps > 0 else 24.0
        ss = range_start / fps if range_start > 0 else None
        dur = total / fps if range_start > 0 else None
        cmd = _build_ffmpeg_cmd_cpu(self._source.file_path, TNET_WIDTH, TNET_HEIGHT)
        if ss is not None:
            # Insert -ss and -t for range-limited decode
            cmd = cmd[:3] + ["-ss", f"{ss:.4f}"] + cmd[3:]
            if dur is not None:
                idx = cmd.index("-i") + 2
                cmd = cmd[:idx] + ["-t", f"{dur + 5:.4f}"] + cmd[idx:]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._procs.append(proc)

        buf = bytearray()
        count = 0
        try:
            while count < total and not self._cancelled:
                raw = proc.stdout.read(frame_size * 500)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= frame_size and count < total:
                    padded[TNET_PAD + count] = (
                        np.frombuffer(bytes(buf[:frame_size]), np.uint8)
                        .reshape(TNET_HEIGHT, TNET_WIDTH, 3)
                    )
                    del buf[:frame_size]
                    count += 1
                    if count % max(1, total // 200) == 0:
                        self.progress.emit(min(100, int(count / total * 100)))
                        self.detail_progress.emit(count, total, "Decoding")
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()

        return count

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
        if self._frame_range:
            range_start, range_end = self._frame_range
            total = range_end - range_start + 1
        else:
            range_start = 0
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
                source_in=range_start + frame_in,
                source_out=range_start + frame_out,
            ))
        return clips
