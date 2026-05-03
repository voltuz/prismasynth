import logging
import time
from enum import Enum
from typing import List, Optional

import av
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.clip import Clip
from core.video_source import VideoSource
from core.proxy_cache import ProxyFile
from core.ffmpeg_decode import decode_to_array

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
FALLBACK_HSV_THRESHOLD = 30.0

# TransNetV2 constants
TNET_WIDTH = 48
TNET_HEIGHT = 27
TNET_WINDOW = 100
TNET_STEP = 50
TNET_PAD = 25


class Detector(Enum):
    """Which cut-detection backend to use."""
    TRANSNETV2 = "transnetv2"
    OMNISHOTCUT = "omnishotcut"


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
                 threshold: float = DEFAULT_THRESHOLD,
                 detector: Detector = Detector.TRANSNETV2,
                 omnishotcut_checkpoint: Optional[str] = None,
                 parent=None):
        """segments: list of (source_id, source_in, source_out, clip_id) tuples.
        sources: dict of source_id -> VideoSource.
        detector: which backend to use (TransNetV2 or OmniShotCut).
        omnishotcut_checkpoint: path to OmniShotCut .pth file (required if detector==OMNISHOTCUT)."""
        super().__init__(parent)
        self._segments = segments
        self._sources = sources
        self._threshold = threshold
        self._detector = detector
        self._omnishotcut_checkpoint = omnishotcut_checkpoint
        self._cancelled = False
        self._procs: list = []
        self._omnishotcut_runner = None  # set when running OmniShotCut

    def cancel(self):
        self._cancelled = True
        # Kill all ffmpeg subprocesses immediately
        for proc in list(self._procs):
            try:
                proc.kill()
            except Exception:
                pass
        # Cancel the OmniShotCut sidecar if running
        if self._omnishotcut_runner is not None:
            try:
                self._omnishotcut_runner.cancel()
            except Exception:
                pass

    def run(self):
        try:
            self._total_all = sum(
                seg[2] - seg[1] + 1 for seg in self._segments
            )
            self._done_all = 0

            if self._detector == Detector.OMNISHOTCUT:
                self._run_omnishotcut()
                return

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

        self.phase_changed.emit("Decoding")

        padded, n_frames, _, _ = decode_to_array(
            source, range_start, total, TNET_WIDTH, TNET_HEIGHT,
            procs=self._procs,
            is_cancelled=lambda: self._cancelled,
            progress_cb=lambda done, _t: self._emit_decode_progress(done),
            pad_before=TNET_PAD,
            pad_after=pad_end,
        )

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

    def _emit_decode_progress(self, seg_done: int):
        overall = self._done_all + seg_done
        pct = int(overall / self._total_all * 100) if self._total_all else 0
        self.progress.emit(min(99, pct))
        self.detail_progress.emit(overall, self._total_all, "Decoding")

    # --- OmniShotCut (sidecar subprocess, all segments in one run) ---

    def _run_omnishotcut(self):
        """Process all segments through the OmniShotCut sidecar.
        Decoding (in-process, GPU-accelerated) and inference (sidecar) are
        interleaved per segment. Emits the same signals as the TransNetV2 path."""
        from core.omnishotcut_runner import OmnishotcutRunner

        if not self._omnishotcut_checkpoint:
            self.error.emit("OmniShotCut checkpoint path not provided.")
            return

        results: dict = {}

        def on_phase(phase: str):
            self.phase_changed.emit(phase)

        def on_decode_progress(seg_done: int):
            self._emit_decode_progress(seg_done)

        def on_analyze_progress(frame_done: int, total: int, seg_id):
            # During inference, OmniShotCut reports per-window progress within
            # the current segment. Map to overall by counting completed segments.
            overall = self._done_all + frame_done
            pct = int(overall / self._total_all * 100) if self._total_all else 0
            self.progress.emit(min(99, pct))
            self.detail_progress.emit(overall, self._total_all, "Analyzing")

        def on_segment_done(clip_id, source: VideoSource, range_start: int,
                            seg_total: int, cuts: List[int]):
            results[clip_id] = self._cuts_to_clips(cuts, source, range_start, seg_total)
            self._done_all += seg_total

        runner = OmnishotcutRunner(
            segments=self._segments,
            sources=self._sources,
            checkpoint_path=self._omnishotcut_checkpoint,
            procs=self._procs,
            is_cancelled=lambda: self._cancelled,
            on_phase=on_phase,
            on_decode_progress=on_decode_progress,
            on_analyze_progress=on_analyze_progress,
            on_segment_done=on_segment_done,
        )
        self._omnishotcut_runner = runner
        try:
            runner.run()
        except Exception as e:
            logger.exception("OmniShotCut run failed")
            self.error.emit(str(e))
            return
        finally:
            self._omnishotcut_runner = None

        if self._cancelled:
            return
        self.finished.emit(results)

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
