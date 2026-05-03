"""Shared ffmpeg-based parallel decode helpers.

Used by SceneDetector (TransNetV2 path) and OmnishotcutRunner (OmniShotCut path)
to decode arbitrary source ranges into a uint8 RGB numpy array at any target
resolution. Tries NVDEC+GPU-scale, then NVDEC+CPU-scale, then single-process
CPU decode in order, falling back when each path's probe fails.
"""

import logging
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.video_source import VideoSource

logger = logging.getLogger(__name__)

# Default number of parallel ffmpeg processes for the fast paths
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
    """GPU-accelerated decode (NVDEC) + CPU scale."""
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
    """Test whether an ffmpeg command produces at least one full frame."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = proc.stdout.read(frame_size * 2)
    proc.stdout.close()
    proc.kill()
    proc.wait()
    return len(data) >= frame_size


IsCancelled = Callable[[], bool]
ProgressCb = Optional[Callable[[int, int], None]]  # (frames_done, total)


def _decode_parallel(padded: np.ndarray, total: int, range_start: int,
                     pad_before: int, source: VideoSource,
                     width: int, height: int,
                     gpu_scale: bool,
                     procs: list, is_cancelled: IsCancelled,
                     progress_cb: ProgressCb) -> int:
    """Decode `total` frames starting at `range_start` using N parallel ffmpeg processes.
    Frames write directly into padded[pad_before:pad_before+total]. Returns count decoded."""
    fps = source.fps if source.fps > 0 else 24.0
    frame_size = width * height * 3
    n_seg = N_DECODE_SEGMENTS
    frames_per_seg = total // n_seg

    build_cmd = _build_ffmpeg_cmd_gpu_scale if gpu_scale else _build_ffmpeg_cmd

    seg_counts = [0] * n_seg
    seg_errors: List[Optional[Exception]] = [None] * n_seg
    progress_lock = threading.Lock()
    total_decoded = [0]

    def decode_segment(seg_idx: int, start_frame: int, max_frames: int):
        ss = start_frame / fps
        dur = (max_frames + 100) / fps
        cmd = build_cmd(source.file_path, width, height, ss=ss, duration=dur)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        procs.append(proc)

        buf = bytearray()
        count = 0
        offset = pad_before + start_frame - range_start
        try:
            while count < max_frames and not is_cancelled():
                raw = proc.stdout.read(frame_size * 500)
                if not raw:
                    break
                buf.extend(raw)
                while len(buf) >= frame_size and count < max_frames:
                    padded[offset + count] = (
                        np.frombuffer(bytes(buf[:frame_size]), np.uint8)
                        .reshape(height, width, 3)
                    )
                    del buf[:frame_size]
                    count += 1
                    if progress_cb is not None:
                        with progress_lock:
                            total_decoded[0] += 1
                            n = total_decoded[0]
                        if n % max(1, total // 200) == 0:
                            progress_cb(n, total)
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

    # Probe the pipeline before spawning workers
    test_cmd = build_cmd(source.file_path, width, height)
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
    errs = [e for e in seg_errors if e is not None]
    if errs:
        logger.warning("Segment decode errors: %s", errs)

    return n_frames


def _decode_single(padded: np.ndarray, total: int, range_start: int,
                   pad_before: int, source: VideoSource,
                   width: int, height: int,
                   procs: list, is_cancelled: IsCancelled,
                   progress_cb: ProgressCb) -> int:
    """Single-process CPU fallback decode."""
    frame_size = width * height * 3
    fps = source.fps if source.fps > 0 else 24.0
    ss = range_start / fps if range_start > 0 else None
    dur = total / fps if range_start > 0 else None
    cmd = _build_ffmpeg_cmd_cpu(source.file_path, width, height)
    if ss is not None:
        cmd = cmd[:3] + ["-ss", f"{ss:.4f}"] + cmd[3:]
        if dur is not None:
            idx = cmd.index("-i") + 2
            cmd = cmd[:idx] + ["-t", f"{dur + 5:.4f}"] + cmd[idx:]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    procs.append(proc)

    buf = bytearray()
    count = 0
    try:
        while count < total and not is_cancelled():
            raw = proc.stdout.read(frame_size * 500)
            if not raw:
                break
            buf.extend(raw)
            while len(buf) >= frame_size and count < total:
                padded[pad_before + count] = (
                    np.frombuffer(bytes(buf[:frame_size]), np.uint8)
                    .reshape(height, width, 3)
                )
                del buf[:frame_size]
                count += 1
                if progress_cb is not None and count % max(1, total // 200) == 0:
                    progress_cb(count, total)
    finally:
        try:
            proc.stdout.close()
            proc.kill()
            proc.wait()
        except Exception:
            pass

    return count


def decode_to_array(
    source: VideoSource,
    range_start: int,
    total: int,
    width: int,
    height: int,
    *,
    procs: list,
    is_cancelled: IsCancelled,
    progress_cb: ProgressCb = None,
    pad_before: int = 0,
    pad_after: int = 0,
) -> Tuple[np.ndarray, int, str, float]:
    """Decode `total` frames starting at `range_start` from `source` at width×height.
    Tries parallel-gpu_scale → parallel-cpu_scale → single-cpu and returns the first
    successful path. Frames are written into a pre-allocated numpy array of shape
    `(pad_before + total + pad_after, height, width, 3)`.

    Returns: (padded_array, n_frames_decoded, method_name, elapsed_secs).
    `method_name` is empty if all paths failed.

    The caller is responsible for tracking `procs` (subprocess.Popen instances are
    appended for cancellation) and providing `is_cancelled` / `progress_cb` callbacks.
    """
    padded = np.empty((pad_before + total + pad_after, height, width, 3), dtype=np.uint8)

    n_frames = 0
    method = ""

    # Try paths in order. _decode_parallel itself probes the pipeline first.
    for label, gpu in (("ffmpeg-parallel-scale_cuda", True),
                       ("ffmpeg-parallel-cpu_scale", False)):
        if is_cancelled():
            return padded, 0, "", 0.0
        t0 = time.monotonic()
        n = _decode_parallel(padded, total, range_start, pad_before, source,
                             width, height, gpu_scale=gpu,
                             procs=procs, is_cancelled=is_cancelled,
                             progress_cb=progress_cb)
        if n:
            elapsed = time.monotonic() - t0
            logger.info("Decode [%s]: %d frames in %.1fs (%.0f fps) at %dx%d",
                        label, n, elapsed, n / max(elapsed, 0.001), width, height)
            return padded, n, label, elapsed

    if is_cancelled():
        return padded, 0, "", 0.0

    t0 = time.monotonic()
    n = _decode_single(padded, total, range_start, pad_before, source,
                      width, height,
                      procs=procs, is_cancelled=is_cancelled,
                      progress_cb=progress_cb)
    if n:
        elapsed = time.monotonic() - t0
        logger.info("Decode [ffmpeg-single-cpu]: %d frames in %.1fs (%.0f fps) at %dx%d",
                    n, elapsed, n / max(elapsed, 0.001), width, height)
        return padded, n, "ffmpeg-single-cpu", elapsed

    logger.warning("All decode paths failed for %s", source.file_path)
    return padded, 0, "", 0.0
