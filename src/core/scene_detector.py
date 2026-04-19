import logging
import subprocess
import threading
import time
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

def _build_ffmpeg_cmd_gpu_scale(file_path: str, width: int, height: int,
                                ss: float = None, duration: float = None) -> List[str]:
    """NVDEC decode + GPU scale via scale_cuda, then hwdownload to pipe."""
    cmd = ["ffmpeg", "-nostdin", "-v", "quiet",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.4f}"]
    cmd += ["-i", file_path]
    if duration is not None:
        cmd += ["-t", f"{duration:.4f}"]
    cmd += ["-vf", f"scale_cuda={width}:{height}:interp_algo=nearest,hwdownload,format=nv12,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def _build_ffmpeg_cmd(file_path: str, width: int, height: int,
                      ss: float = None, duration: float = None) -> List[str]:
    """Build ffmpeg decode command with GPU-accelerated decode + CPU scale."""
    cmd = ["ffmpeg", "-nostdin", "-v", "quiet", "-hwaccel", "cuda"]
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
        "ffmpeg", "-nostdin", "-v", "quiet", "-threads", "0",
        "-i", file_path,
        "-vf", f"scale={width}:{height}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def _probe_ffmpeg_cmd(cmd: List[str], frame_size: int) -> bool:
    """Test if an ffmpeg command produces valid output frames."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = proc.stdout.read(frame_size * 2)
    proc.stdout.close()
    proc.kill()
    proc.wait()
    return len(data) >= frame_size


class SceneDetector(QThread):
    """Detects shot boundaries using TransNetV2 (GPU-accelerated neural network).
    Accepts a list of segments (source, source_in, source_out, clip_id) and processes
    each independently. Uses parallel NVDEC decoding for speed. Falls back to HSV
    if TransNetV2 unavailable."""

    progress = Signal(int)            # percent (0-100)
    detail_progress = Signal(int, int, str)  # frames_done, total_frames, phase
    phase_changed = Signal(str)       # emitted when switching stages
    finished = Signal(dict)           # {clip_id: [Clip, ...]}
    error = Signal(str)

    def __init__(self, segments: list, sources: dict,
                 threshold: float = DEFAULT_THRESHOLD, parent=None):
        """segments: list of (source_id, source_in, source_out, clip_id) tuples.
        sources: dict of source_id -> VideoSource."""
        super().__init__(parent)
        self._segments = segments
        self._sources = sources
        self._threshold = threshold
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
            # Calculate total frames across all segments for overall progress
            self._total_all = sum(
                seg[2] - seg[1] + 1 for seg in self._segments
            )
            self._done_all = 0

            results = {}  # clip_id -> [Clip, ...]

            use_transnet = self._check_transnet()
            if use_transnet is None and not self._cancelled:
                logger.info("TransNetV2 unavailable, falling back to HSV method")

            for source_id, source_in, source_out, clip_id in self._segments:
                if self._cancelled:
                    return
                source = self._sources[source_id]
                self._current_source = source
                self._current_range = (source_in, source_out)
                seg_total = source_out - source_in + 1

                if use_transnet is not None:
                    cuts = self._detect_segment_transnet(
                        use_transnet, source, source_in, seg_total
                    )
                else:
                    cuts = self._detect_segment_hsv(source, source_in, seg_total)

                if self._cancelled:
                    return

                clips = self._cuts_to_clips(cuts, source, source_in, seg_total)
                results[clip_id] = clips
                self._done_all += seg_total

            self.finished.emit(results)
        except Exception as e:
            logger.exception("Scene detection failed")
            self.error.emit(str(e))

    def _check_transnet(self):
        """Try to load TransNetV2 model. Returns model or None."""
        try:
            import torch
            from transnetv2_pytorch import TransNetV2
            model = TransNetV2(device="auto")
            logger.info("TransNetV2 loaded on %s", model.device)
            return model
        except ImportError:
            return None
        except Exception as e:
            logger.warning("TransNetV2 failed to initialize: %s", e)
            return None

    # --- TransNetV2 (per-segment) ---

    def _detect_segment_transnet(self, model, source: VideoSource,
                                 range_start: int, total: int) -> List[int]:
        import torch

        pad_end = TNET_PAD + TNET_STEP - (total % TNET_STEP if total % TNET_STEP != 0 else TNET_STEP)

        # Pre-allocate padded array — all decode paths write directly into it
        padded = np.empty((TNET_PAD + total + pad_end, TNET_HEIGHT, TNET_WIDTH, 3), dtype=np.uint8)

        self.phase_changed.emit("Decoding")

        # Decode fallback chain — each returns 0 on failure
        n_frames = 0
        decode_method = None

        if not n_frames:
            t0 = time.monotonic()
            n_frames = self._decode_parallel(padded, total, range_start, source,
                                             gpu_scale=True)
            if n_frames:
                decode_method = "ffmpeg-parallel-scale_cuda"
                elapsed = time.monotonic() - t0
                logger.info("Decode [ffmpeg-parallel-scale_cuda]: %d frames in %.1fs (%.0f fps)",
                            n_frames, elapsed, n_frames / max(elapsed, 0.001))

        if not n_frames:
            t0 = time.monotonic()
            n_frames = self._decode_parallel(padded, total, range_start, source,
                                             gpu_scale=False)
            if n_frames:
                decode_method = "ffmpeg-parallel-cpu_scale"
                elapsed = time.monotonic() - t0
                logger.info("Decode [ffmpeg-parallel-cpu_scale]: %d frames in %.1fs (%.0f fps)",
                            n_frames, elapsed, n_frames / max(elapsed, 0.001))

        if not n_frames:
            t0 = time.monotonic()
            n_frames = self._decode_single(padded, total, range_start, source)
            if n_frames:
                decode_method = "ffmpeg-single-cpu"
                elapsed = time.monotonic() - t0
                logger.info("Decode [ffmpeg-single-cpu]: %d frames in %.1fs (%.0f fps)",
                            n_frames, elapsed, n_frames / max(elapsed, 0.001))

        if not decode_method:
            logger.warning("All decode paths failed for %s", source.file_path)

        if self._cancelled or n_frames == 0:
            return []

        # Pad edges
        padded[:TNET_PAD] = padded[TNET_PAD]
        padded[TNET_PAD + n_frames:] = padded[TNET_PAD + n_frames - 1]

        if self._cancelled:
            return []

        # Save proxy with frame offset so thumbnail cache can map source
        # frame numbers to proxy indices regardless of where decode started.
        frames_list = [padded[TNET_PAD + i] for i in range(n_frames)]
        ProxyFile.save_frames(source, frames_list, frame_offset=range_start)
        del frames_list

        if self._cancelled:
            return []

        logger.info("Decoded %d frames from %s, running inference",
                     n_frames, source.file_path)
        self.phase_changed.emit("Analyzing")
        t_infer = time.monotonic()

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
                    seg_done = min(n_frames, window_idx * TNET_STEP)
                    overall = self._done_all + seg_done
                    pct = int(overall / self._total_all * 100)
                    self.progress.emit(min(99, pct))
                    self.detail_progress.emit(overall, self._total_all, "Analyzing")

        all_preds = torch.cat(predictions, 0)[:n_frames].numpy()
        cut_frames = np.where(all_preds > self._threshold)[0].tolist()

        infer_elapsed = time.monotonic() - t_infer
        logger.info("Inference [TransNetV2]: %d frames in %.1fs (%.0f fps) — %d cuts (threshold=%.2f)",
                     n_frames, infer_elapsed, n_frames / max(infer_elapsed, 0.001),
                     len(cut_frames), self._threshold)
        return cut_frames

    # --- Decode: parallel ffmpeg segments ---

    def _decode_parallel(self, padded: np.ndarray, total: int,
                         range_start: int, source: VideoSource,
                         gpu_scale: bool = False) -> int:
        """Decode using N parallel ffmpeg/NVDEC processes.
        gpu_scale=True uses scale_cuda (resize on GPU), False uses CPU scale."""
        fps = source.fps if source.fps > 0 else 24.0
        frame_size = TNET_WIDTH * TNET_HEIGHT * 3
        n_seg = N_DECODE_SEGMENTS
        frames_per_seg = total // n_seg

        # Pick the command builder
        build_cmd = _build_ffmpeg_cmd_gpu_scale if gpu_scale else _build_ffmpeg_cmd

        seg_counts = [0] * n_seg
        seg_errors = [None] * n_seg
        progress_lock = threading.Lock()
        total_decoded = [0]

        def decode_segment(seg_idx, start_frame, max_frames):
            ss = start_frame / fps
            dur = (max_frames + 100) / fps
            cmd = build_cmd(source.file_path, TNET_WIDTH, TNET_HEIGHT,
                            ss=ss, duration=dur)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._procs.append(proc)

            buf = bytearray()
            count = 0
            offset = TNET_PAD + start_frame - range_start
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
                            overall = self._done_all + n
                            pct = int(overall / self._total_all * 100)
                            self.progress.emit(min(99, pct))
                            self.detail_progress.emit(overall, self._total_all, "Decoding")
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

        # Test if this pipeline works with a quick probe
        test_cmd = build_cmd(source.file_path, TNET_WIDTH, TNET_HEIGHT)
        if not _probe_ffmpeg_cmd(test_cmd, frame_size):
            label = "scale_cuda" if gpu_scale else "NVDEC"
            logger.info("ffmpeg %s unavailable, trying next path", label)
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
                       range_start: int, source: VideoSource) -> int:
        """Single ffmpeg process decode — CPU fallback."""
        frame_size = TNET_WIDTH * TNET_HEIGHT * 3
        fps = source.fps if source.fps > 0 else 24.0
        ss = range_start / fps if range_start > 0 else None
        dur = total / fps if range_start > 0 else None
        cmd = _build_ffmpeg_cmd_cpu(source.file_path, TNET_WIDTH, TNET_HEIGHT)
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
                        overall = self._done_all + count
                        pct = int(overall / self._total_all * 100)
                        self.progress.emit(min(99, pct))
                        self.detail_progress.emit(overall, self._total_all, "Decoding")
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()

        return count

    # --- HSV fallback (per-segment) ---

    def _detect_segment_hsv(self, source: VideoSource,
                            range_start: int, total: int) -> List[int]:
        """Fallback: detect cuts using HSV frame differencing."""
        container = av.open(source.file_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        cuts: List[int] = []
        prev_hsv = None
        frame_index = 0
        range_end = range_start + total
        hsv_threshold = FALLBACK_HSV_THRESHOLD

        for packet in container.demux(stream):
            if self._cancelled:
                break
            for frame in packet.decode():
                if self._cancelled:
                    break
                if frame_index < range_start:
                    frame_index += 1
                    continue
                if frame_index >= range_end:
                    break
                rgb = frame.to_ndarray(format="rgb24")
                small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
                if prev_hsv is not None:
                    diff = np.mean(np.abs(hsv - prev_hsv))
                    if diff > hsv_threshold:
                        cuts.append(frame_index - range_start)
                prev_hsv = hsv
                frame_index += 1
                seg_done = frame_index - range_start
                if seg_done % max(1, total // 100) == 0:
                    overall = self._done_all + seg_done
                    pct = int(overall / self._total_all * 100)
                    self.progress.emit(min(99, pct))
            if frame_index >= range_end:
                break

        container.close()
        logger.info("HSV fallback detected %d cuts in %s", len(cuts), source.file_path)
        return cuts

    # --- Common ---

    def _cuts_to_clips(self, cuts: List[int], source: VideoSource,
                       range_start: int, total: int) -> List[Clip]:
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
                source_id=source.id,
                source_in=range_start + frame_in,
                source_out=range_start + frame_out,
            ))
        return clips
