import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import av
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from core.proxy_cache import ProxyManager
from core.timeline import TimelineModel
from core.video_source import VideoSource
from utils.diag import diag
from utils.ffprobe import probe_hdr
from utils.paths import get_cache_dir

logger = logging.getLogger(__name__)

THUMB_WIDTH = 192
THUMB_HEIGHT = 108
THUMB_FRAME_SIZE = THUMB_WIDTH * THUMB_HEIGHT * 3
_MAX_WORKERS = 6

# Bulk-decode parallelism. Two pools because the two paths scale very
# differently:
#   - LIGHT (PyAV sweep, single-threaded decode per container) is bound
#     by Windows file-handle contention + per-chunk open/seek overhead,
#     not by CPU. Empirical sweep on a 32-thread machine showed the
#     throughput peak is at 8 workers; 12 is comparable; 16+ regresses
#     because too many simultaneous av.open() calls thrash the kernel.
#   - HEAVY (per-frame NVDEC ffmpeg subprocess) is GPU-bound. Consumer
#     NVIDIA cards typically allow 5-8 concurrent NVDEC sessions; 6 is
#     the safe baseline.
_BULK_LIGHT_WORKERS = max(4, min(os.cpu_count() or 6, 8))
_BULK_HEAVY_WORKERS = 6


def _build_bulk_cmd_gpu_scale(file_path: str, timestamp: float) -> list:
    """NVDEC decode + GPU scale via scale_cuda. Fastest for HEVC/HDR sources.
    Mirrors scene_detector._build_ffmpeg_cmd_gpu_scale."""
    return [
        "ffmpeg", "-nostdin", "-v", "quiet",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{timestamp:.4f}",
        "-i", file_path,
        "-frames:v", "1",
        "-vf",
        f"scale_cuda={THUMB_WIDTH}:{THUMB_HEIGHT}:format=nv12,"
        "hwdownload,format=nv12,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def _build_bulk_cmd_gpu_decode(file_path: str, timestamp: float) -> list:
    """NVDEC decode + CPU scale. Fallback when scale_cuda is unavailable
    (older drivers / unsupported pixel formats)."""
    return [
        "ffmpeg", "-nostdin", "-v", "quiet",
        "-hwaccel", "cuda",
        "-ss", f"{timestamp:.4f}",
        "-i", file_path,
        "-frames:v", "1",
        "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def _build_bulk_cmd_cpu(file_path: str, timestamp: float) -> list:
    """Pure-CPU fallback. Used only when no NVDEC path probes successfully
    (no NVIDIA GPU, missing CUDA driver, or codec unsupported by NVDEC)."""
    return [
        "ffmpeg", "-nostdin", "-v", "quiet",
        "-ss", f"{timestamp:.4f}",
        "-i", file_path,
        "-frames:v", "1",
        "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def _probe_decode_cmd(cmd: list) -> bool:
    """Run an ffmpeg cmd briefly and check it produces a full thumbnail
    frame's worth of bytes. Mirrors scene_detector._probe_ffmpeg_cmd."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            data = proc.stdout.read(THUMB_FRAME_SIZE)
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        return False
    return len(data) >= THUMB_FRAME_SIZE


# If the next target is within this many frames, keep decoding forward
# instead of re-seeking. Roughly 2x typical H.265 GOP size.
_SEQUENTIAL_THRESHOLD = 400


class ThumbnailCache(QObject):
    """Long-lived thumbnail generator. Created once, survives timeline edits.

    Uses a sweep strategy: from the playhead, sweeps right (forward) and left
    (backward) through sorted frame targets. Within each sweep, sequential
    decode avoids redundant re-seeks for nearby frames. Distant jumps fall
    back to independent seeks.
    """

    thumbnail_ready = Signal(str, str, QImage)  # clip_id, "first"|"last", qimage

    def __init__(self, timeline: TimelineModel,
                 sources: Dict[str, VideoSource],
                 proxy_manager: ProxyManager = None, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._sources = sources
        self._proxy_manager = proxy_manager
        self._mem_cache: Dict[str, QImage] = {}
        self._lq_emitted: Set[str] = set()
        self._stopped = False
        self._coord_thread: Optional[threading.Thread] = None
        self._pool: Optional[ThreadPoolExecutor] = None

        # Pause support (scrubbing priority)
        self._pause_event = threading.Event()
        self._pause_event.set()

        # Clip lookup — rebuilt on notify_clips_changed()
        self._clip_lookup: Dict[str, List[Tuple[str, int, str]]] = {}
        self._lookup_lock = threading.Lock()

        # Priority management — updated from main thread on scroll
        self._priority_clip_ids: Set[str] = set()
        self._playhead_frame: int = 0
        self._priority_lock = threading.Lock()

        # Wake event — signals coordinator to re-check for work
        self._wake_event = threading.Event()

        # When False, coordinator still emits LQ proxy placeholders but skips
        # submitting expensive HQ ffmpeg/PyAV decode jobs.
        self._hq_enabled = True

        # Track live subprocesses so stop() can kill them immediately
        # instead of waiting for per-frame communicate() timeouts.
        self._active_procs: Set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()

        # Bulk decode method per source-id, populated lazily by
        # _get_bulk_decode_method on first use. Cached for the cache's
        # lifetime since the source file's codec doesn't change. Value is
        # one of the _build_bulk_cmd_* functions or None for "no method
        # works for this source".
        self._bulk_decode: Dict[str, Optional[callable]] = {}
        self._bulk_decode_lock = threading.Lock()

        # Set while a BulkCacheJob is running. The coordinator's
        # _submit_sweeps short-circuits when this is set so it doesn't
        # spawn parallel PyAV containers on the same source — preventing
        # native-side resource exhaustion (file handles + libav decode
        # threads) that crashed Run 1 of the Fantasy Island bake.
        self._bulk_in_progress = threading.Event()

    def set_hq_enabled(self, enabled: bool):
        """Toggle HQ thumbnail generation. LQ proxy placeholders keep emitting
        either way while the cache is running."""
        self._hq_enabled = enabled
        self._wake_event.set()

    def pause(self):
        self._pause_event.clear()
        # Kill in-flight ffmpeg subprocesses so single-frame grabs abort
        # immediately instead of holding CPU/disk through communicate().
        with self._procs_lock:
            procs = list(self._active_procs)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    def resume(self):
        self._pause_event.set()
        self._wake_event.set()

    def start(self, priority_clip_ids: Set[str] = None,
              playhead_frame: int = 0):
        """Start the persistent coordinator thread if not already running."""
        with self._priority_lock:
            self._priority_clip_ids = set(priority_clip_ids or [])
            self._playhead_frame = playhead_frame
        self._rebuild_clip_lookup()
        if self._coord_thread and self._coord_thread.is_alive():
            self._wake_event.set()
            return
        self._stopped = False
        self._pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        self._coord_thread = threading.Thread(target=self._coordinate, daemon=True)
        self._coord_thread.start()

    def notify_clips_changed(self):
        """Called when clips change (split, detection, import, delete).
        Rebuilds the clip lookup without destroying the cache or thread pool."""
        self._rebuild_clip_lookup()
        self._wake_event.set()

    def _rebuild_clip_lookup(self):
        lookup: Dict[str, List[Tuple[str, int, str]]] = {}
        for clip in self._timeline.clips:
            if clip.is_gap:
                continue
            lookup[clip.id] = [
                (clip.source_id, clip.source_in, "first"),
                (clip.source_id, clip.source_out, "last"),
            ]
        with self._lookup_lock:
            self._clip_lookup = lookup

    def reprioritize(self, visible_clip_ids: Set[str],
                     playhead_frame: int = 0):
        """Called on scroll — next submission uses the new viewport."""
        with self._priority_lock:
            self._priority_clip_ids = set(visible_clip_ids)
            self._playhead_frame = playhead_frame
        self._wake_event.set()

    def _coordinate(self):
        """Persistent coordinator — plans sweep batches, dispatches to pool."""
        in_flight: Dict[Future, List[Tuple[str, int, str, str]]] = {}

        try:
            while not self._stopped:
                # Wait if paused (scrubbing)
                while not self._pause_event.wait(timeout=0.1):
                    if self._stopped:
                        return

                # Drain completed futures and emit results
                done = [f for f in in_flight if f.done()]
                for f in done:
                    entries = in_flight.pop(f)
                    try:
                        results = f.result()  # list of (frame_num, QImage)
                    except Exception:
                        continue
                    if results is None:
                        continue
                    # Map results back to clip_id/position for emission
                    saved_to_disk: Set[Tuple[str, int]] = set()
                    for frame_num, qimage in results:
                        for source_id, fn, clip_id, position in entries:
                            if fn == frame_num:
                                cache_key = f"{source_id}_{frame_num}"
                                self._mem_cache[cache_key] = qimage
                                self.thumbnail_ready.emit(clip_id, position, qimage)
                                # Persist once per (source, frame), not once
                                # per emission (multiple clips may reference
                                # the same source frame).
                                key = (source_id, frame_num)
                                if key not in saved_to_disk:
                                    self._save_disk_thumbnail(
                                        source_id, frame_num, qimage)
                                    saved_to_disk.add(key)

                # Plan and submit sweep batches
                if len(in_flight) < _MAX_WORKERS and not self._stopped:
                    self._submit_sweeps(in_flight)

                if not in_flight:
                    self._wake_event.wait(timeout=1.0)
                    self._wake_event.clear()
                else:
                    time.sleep(0.01)

        finally:
            if self._pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None

    def _submit_sweeps(self, in_flight):
        """Build sweep batches from visible frames and submit to pool."""
        # Stand down while a bulk bake is running — multiple PyAV containers
        # on the same source from coordinator + bulk pools simultaneously
        # was the root cause of the silent native crash on the Fantasy
        # Island reproducer. The bulk job calls _wake_event.set() when it
        # finishes so the coordinator picks back up immediately.
        if self._bulk_in_progress.is_set():
            return
        with self._priority_lock:
            visible_ids = set(self._priority_clip_ids)
            playhead = self._playhead_frame

        with self._lookup_lock:
            clip_lookup = dict(self._clip_lookup)

        # Already in-flight frame keys
        in_flight_keys = set()
        for entries in in_flight.values():
            for source_id, frame_num, clip_id, position in entries:
                in_flight_keys.add((source_id, frame_num))

        # Collect all uncached visible frames, grouped by source
        by_source: Dict[str, List[Tuple[int, str, str]]] = {}
        for clip_id in visible_ids:
            clip_entries = clip_lookup.get(clip_id)
            if not clip_entries:
                continue
            for source_id, frame_num, position in clip_entries:
                if (source_id, frame_num) in in_flight_keys:
                    continue
                cache_key = f"{source_id}_{frame_num}"
                if cache_key in self._mem_cache:
                    self.thumbnail_ready.emit(
                        clip_id, position, self._mem_cache[cache_key])
                    continue
                # Disk hit: load JPEG, populate mem cache, emit. Skips both
                # LQ placeholder and HQ generation entirely.
                disk_qimg = self._load_disk_thumbnail(source_id, frame_num)
                if disk_qimg is not None:
                    self._mem_cache[cache_key] = disk_qimg
                    self.thumbnail_ready.emit(clip_id, position, disk_qimg)
                    continue
                # Emit LQ placeholder
                if (cache_key not in self._lq_emitted
                        and self._proxy_manager is not None):
                    proxy = self._proxy_manager.get_proxy(source_id)
                    if proxy:
                        lq_frame = proxy.get_frame(frame_num)
                        if lq_frame is not None:
                            thumb = cv2.resize(
                                lq_frame, (THUMB_WIDTH, THUMB_HEIGHT),
                                interpolation=cv2.INTER_NEAREST)
                            h, w, ch = thumb.shape
                            qimg = QImage(bytes(thumb.data), w, h, ch * w,
                                          QImage.Format.Format_RGB888).copy()
                            self.thumbnail_ready.emit(clip_id, position, qimg)
                            self._lq_emitted.add(cache_key)
                by_source.setdefault(source_id, []).append(
                    (frame_num, clip_id, position))

        if not by_source:
            return

        # LQ placeholders were emitted above; skip the expensive HQ decode
        # jobs when the user has disabled HQ generation.
        if not self._hq_enabled:
            return

        # For each source, build sweep runs ordered by playhead distance
        for source_id, frames in by_source.items():
            source = self._sources.get(source_id)
            if source is None:
                continue

            # Split into right sweep (>= playhead) and left sweep (< playhead)
            # Each sorted by frame number ascending for sequential decode
            right = sorted([f for f in frames if f[0] >= playhead], key=lambda x: x[0])
            left = sorted([f for f in frames if f[0] < playhead], key=lambda x: x[0])

            # Build runs within each sweep (group frames within SEQUENTIAL_THRESHOLD)
            for sweep_frames in [right, left]:
                runs = self._build_runs(sweep_frames)
                for run in runs:
                    if len(in_flight) >= _MAX_WORKERS:
                        return
                    target_frames = [f[0] for f in run]
                    entries = [(source_id, f[0], f[1], f[2]) for f in run]
                    # Check none are already in-flight
                    if any((source_id, fn) in in_flight_keys for fn in target_frames):
                        continue
                    for fn in target_frames:
                        in_flight_keys.add((source_id, fn))

                    if len(target_frames) == 1:
                        # Single frame — use fast independent seek
                        fut = self._pool.submit(
                            self._grab_frame_single, source, target_frames[0])
                        in_flight[fut] = entries
                    else:
                        # Multiple frames — sequential sweep decode
                        fut = self._pool.submit(
                            self._grab_frames_sweep, source, target_frames)
                        in_flight[fut] = entries

    @staticmethod
    def _build_runs(frames: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
        """Group sorted frames into runs where consecutive frames are within
        SEQUENTIAL_THRESHOLD of each other."""
        if not frames:
            return []
        runs = [[frames[0]]]
        for f in frames[1:]:
            if f[0] - runs[-1][-1][0] <= _SEQUENTIAL_THRESHOLD:
                runs[-1].append(f)
            else:
                runs.append([f])
        return runs

    @staticmethod
    def _is_heavy_source(source: VideoSource) -> bool:
        """Pick the right bulk-decode path for this source.

        Light = H.264 SDR ≤1080p — PyAV sweep handles it fast and reliably;
        amortizes one decoder context across many target frames per call.

        Heavy = HEVC, AV1, ≥4K, or HDR — software libav decode is either
        slow (4K/HEVC) or unreliable (Dolby Vision metadata can fail
        decode entirely, observed crash on Monster S01E08). Each frame
        becomes its own ffmpeg subprocess with NVDEC, so a libav segfault
        on one frame can't take down the others.
        """
        codec = (source.codec or "").lower()
        if codec not in ("h264", "avc", "avc1"):
            return True
        if source.width >= 3000 or source.height >= 1800:
            return True
        # HDR detection requires its own ffprobe call (cached in
        # ffprobe._hdr_cache after first lookup, so cheap on repeat).
        if probe_hdr(source.file_path):
            return True
        return False

    def _get_bulk_decode_method(self, source: VideoSource):
        """Return a `_build_bulk_cmd_*` builder appropriate for this source,
        probing the NVDEC → NVDEC-decode → CPU cascade once per source-id.
        Returns None if no method produces output (very rare: missing
        ffmpeg / corrupt file). Cached across the cache's lifetime."""
        with self._bulk_decode_lock:
            if source.id in self._bulk_decode:
                return self._bulk_decode[source.id]
        # Probe outside the lock so concurrent first-use on different sources
        # doesn't serialize. Worst case two threads probe the same source
        # simultaneously and both write the same result.
        # Probe at frame 0 — codec capability is per-stream, not per-frame.
        diag(f"bulk    probe sid={source.id[:6]} {source.codec} "
             f"{source.width}x{source.height}")
        for builder, label in (
            (_build_bulk_cmd_gpu_scale, "scale_cuda"),
            (_build_bulk_cmd_gpu_decode, "nvdec+cpu_scale"),
            (_build_bulk_cmd_cpu, "cpu"),
        ):
            cmd = builder(source.file_path, 0.0)
            if _probe_decode_cmd(cmd):
                diag(f"bulk    probe-ok sid={source.id[:6]} method={label}")
                with self._bulk_decode_lock:
                    self._bulk_decode[source.id] = builder
                return builder
        diag(f"bulk    probe-FAIL sid={source.id[:6]}")
        with self._bulk_decode_lock:
            self._bulk_decode[source.id] = None
        return None

    def _bulk_worker_light(self, source: VideoSource, source_id: str,
                           chunk_frames: List[int]) -> int:
        """Worker function for the LIGHT bulk path. Decodes a chunk, then
        does ALL post-processing (disk save + mem-cache + signal emit) on
        the worker thread itself. Returns frame count for progress.

        Moving post-processing into the worker is what lets the bulk pool
        scale past ~8 workers. Otherwise the orchestrator thread serializes
        ~3-5 ms of save+emit per frame, which becomes the wall-time floor.
        """
        results = self._grab_frames_sweep(source, chunk_frames)
        if results:
            for frame_num, qimage in results:
                cache_key = f"{source_id}_{frame_num}"
                self._mem_cache[cache_key] = qimage
                ThumbnailCache._save_disk_thumbnail(
                    source_id, frame_num, qimage)
                self.emit_for_frame(source_id, frame_num, qimage)
        return len(chunk_frames)

    def _bulk_worker_heavy(self, source: VideoSource, source_id: str,
                           frame_num: int) -> int:
        """Worker function for the HEAVY bulk path. Same parallel-post-
        processing pattern as _bulk_worker_light. Returns 1 for progress."""
        results = self._grab_frame_bulk(source, frame_num)
        if results:
            for fn, qimage in results:
                cache_key = f"{source_id}_{fn}"
                self._mem_cache[cache_key] = qimage
                ThumbnailCache._save_disk_thumbnail(source_id, fn, qimage)
                self.emit_for_frame(source_id, fn, qimage)
        return 1

    def _grab_frame_bulk(self, source: VideoSource,
                         frame_num: int) -> Optional[List[Tuple[int, QImage]]]:
        """Grab one frame via the per-source-cached bulk decode method.
        Each call is a fully isolated ffmpeg subprocess: a crash in one
        subprocess (libav segfault, NVDEC error) doesn't affect other
        workers. Used by BulkCacheJob; not used by the viewport-driven
        coordinator (which intentionally avoids GPU contention with mpv)."""
        if self._stopped or not self._pause_event.is_set():
            return None
        builder = self._get_bulk_decode_method(source)
        if builder is None:
            return None
        timestamp = frame_num / source.fps if source.fps > 0 else 0.0
        cmd = builder(source.file_path, timestamp)
        diag(f"bulk    grab sid={source.id[:6]} f={frame_num}")
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self._procs_lock:
                if self._stopped:
                    proc.kill()
                    return None
                self._active_procs.add(proc)
            try:
                data, _ = proc.communicate(timeout=30)
            finally:
                with self._procs_lock:
                    self._active_procs.discard(proc)
        except (OSError, subprocess.TimeoutExpired) as e:
            diag(f"bulk    grab-ERR sid={source.id[:6]} f={frame_num} "
                 f"{type(e).__name__}")
            if proc:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            return None

        if self._stopped or len(data) < THUMB_FRAME_SIZE:
            diag(f"bulk    grab-empty sid={source.id[:6]} f={frame_num} "
                 f"bytes={len(data) if data else 0}")
            return None

        rgb = np.frombuffer(data[:THUMB_FRAME_SIZE], dtype=np.uint8).reshape(
            THUMB_HEIGHT, THUMB_WIDTH, 3)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()
        return [(frame_num, qimg)]

    def _grab_frame_single(self, source: VideoSource,
                           frame_num: int) -> Optional[List[Tuple[int, QImage]]]:
        """Grab one frame via ffmpeg seek. Used for isolated frames."""
        if self._stopped or not self._pause_event.is_set():
            return None
        diag(f"single  enter sid={source.id[:6]} f={frame_num}")
        timestamp = frame_num / source.fps if source.fps > 0 else 0.0
        cmd = [
            "ffmpeg", "-nostdin", "-v", "quiet",
            "-ss", f"{timestamp:.4f}",
            "-i", source.file_path,
            "-frames:v", "1",
            "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = None
        try:
            diag(f"single  popen f={frame_num}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            diag(f"single  popen-done f={frame_num} pid={proc.pid}")
            with self._procs_lock:
                if self._stopped:
                    proc.kill()
                    return None
                self._active_procs.add(proc)
            try:
                diag(f"single  communicate f={frame_num} pid={proc.pid}")
                data, _ = proc.communicate(timeout=15)
                diag(f"single  comm-done   f={frame_num} pid={proc.pid} bytes={len(data) if data else 0}")
            finally:
                with self._procs_lock:
                    self._active_procs.discard(proc)
        except (OSError, subprocess.TimeoutExpired) as e:
            diag(f"single  ERR f={frame_num} {type(e).__name__}: {e}")
            if proc:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
            return None

        if self._stopped or len(data) != THUMB_FRAME_SIZE:
            return None

        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            THUMB_HEIGHT, THUMB_WIDTH, 3)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb.data), w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()
        return [(frame_num, qimg)]

    def _grab_frames_sweep(self, source: VideoSource,
                           target_frames: List[int]) -> Optional[List[Tuple[int, QImage]]]:
        """Grab multiple frames via a single PyAV sequential decode.
        target_frames must be sorted ascending. Seeks once to the first target,
        then decodes forward through all targets."""
        if self._stopped or not self._pause_event.is_set():
            return None
        diag(f"sweep   enter sid={source.id[:6]} n={len(target_frames)} "
             f"first={target_frames[0]} last={target_frames[-1]}")
        fps = source.fps if source.fps > 0 else 24.0
        results = []
        try:
            diag(f"sweep   av.open sid={source.id[:6]}")
            container = av.open(source.file_path)
            diag(f"sweep   av.open-done sid={source.id[:6]}")
            stream = container.streams.video[0]
            # Single-threaded decode per container. Pool-level parallelism
            # already provides throughput; AUTO on top of 6 parallel
            # containers blew past native resource limits and silently
            # segfaulted on the 1153-frame Fantasy Island bake (see plan:
            # "Bulk Cache crash on light-path bake").
            stream.thread_type = "NONE"
            tb = float(stream.time_base)

            targets = list(target_frames)
            target_pts = [int(f / fps / tb) for f in targets]
            idx = 0  # next target to find

            # Seek to just before first target
            diag(f"sweep   seek sid={source.id[:6]} pts={target_pts[0]}")
            container.seek(target_pts[0], stream=stream)
            diag(f"sweep   seek-done sid={source.id[:6]}")

            for frame in container.decode(stream):
                # Abort mid-sweep when the user is scrubbing — frees CPU
                # for the Qt main thread to repaint the playhead.
                if self._stopped or not self._pause_event.is_set():
                    break
                if frame.pts is None:
                    continue
                # Check if we've reached or passed the current target
                while idx < len(targets) and frame.pts >= target_pts[idx]:
                    rgb = frame.to_ndarray(format="rgb24")
                    thumb = cv2.resize(rgb, (THUMB_WIDTH, THUMB_HEIGHT),
                                       interpolation=cv2.INTER_AREA)
                    h, w, ch = thumb.shape
                    qimg = QImage(bytes(thumb.data), w, h, ch * w,
                                  QImage.Format.Format_RGB888).copy()
                    results.append((targets[idx], qimg))
                    idx += 1
                if idx >= len(targets):
                    break

            container.close()
            diag(f"sweep   exit sid={source.id[:6]} got={len(results)}")
        except Exception as e:
            diag(f"sweep   ERR sid={source.id[:6]} {type(e).__name__}: {e}")
            logger.debug("Sweep decode error: %s", e)

        return results if results else None

    def stop(self):
        self._stopped = True
        self._pause_event.set()
        self._wake_event.set()
        # Kill any in-flight ffmpeg subprocesses so workers return immediately
        # instead of blocking on communicate().
        with self._procs_lock:
            procs = list(self._active_procs)
            self._active_procs.clear()
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        if self._coord_thread and self._coord_thread.is_alive():
            self._coord_thread.join(timeout=3.0)
        self._coord_thread = None
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def invalidate_source(self, source_id: str):
        """Remove all cached thumbnails for a source so they regenerate."""
        keys = [k for k in self._mem_cache if k.startswith(f"{source_id}_")]
        for k in keys:
            del self._mem_cache[k]
        lq_keys = [k for k in self._lq_emitted if k.startswith(f"{source_id}_")]
        for k in lq_keys:
            self._lq_emitted.discard(k)
        self._wake_event.set()

    def clear(self):
        self._mem_cache.clear()
        self._lq_emitted.clear()

    def emit_for_frame(self, source_id: str, frame_num: int,
                       qimage: QImage):
        """Emit thumbnail_ready for every clip currently referencing this
        (source_id, frame_num). Used by bulk-cache to surface results
        without waiting for the next coordinator sweep."""
        with self._lookup_lock:
            clip_lookup = dict(self._clip_lookup)
        for clip_id, entries in clip_lookup.items():
            for sid, fn, position in entries:
                if sid == source_id and fn == frame_num:
                    self.thumbnail_ready.emit(clip_id, position, qimage)

    def start_bulk_cache(self,
                         render_range: Optional[Tuple[int, int]] = None,
                         force_overwrite: bool = False,
                         ) -> "BulkCacheJob":
        """Build a list of (source_id, frame_num) targets for the first and
        last frame of every non-gap clip within render_range (inclusive)
        and start a background BulkCacheJob to fill them. By default frames
        already in memory or on disk are skipped; pass force_overwrite=True
        to queue every in-scope frame for re-decode (existing JPEGs are
        overwritten in place by the bulk job's write-on-result path)."""
        # Make sure emit_for_frame can find clips even if start() was never
        # called (e.g. master thumbnails toggle is off).
        self._rebuild_clip_lookup()
        targets: List[Tuple[str, int]] = []
        seen: Set[Tuple[str, int]] = set()
        pos = 0
        for clip in self._timeline.clips:
            clip_start = pos
            clip_end = pos + clip.duration_frames - 1
            pos += clip.duration_frames
            if clip.is_gap:
                continue
            if render_range is not None:
                r_start, r_end = render_range
                if clip_end < r_start or clip_start > r_end:
                    continue
            for frame_num in (clip.source_in, clip.source_out):
                key = (clip.source_id, frame_num)
                if key in seen:
                    continue
                seen.add(key)
                if not force_overwrite:
                    cache_key = f"{clip.source_id}_{frame_num}"
                    if cache_key in self._mem_cache:
                        continue
                    if self._disk_thumb_path(clip.source_id,
                                             frame_num).exists():
                        continue
                targets.append(key)
        job = BulkCacheJob(self, targets)
        return job

    def disk_thumbnail_stats(self, source_ids: Iterable[str]
                             ) -> Tuple[int, int]:
        """Return (file_count, total_bytes) of cached JPEGs on disk for
        the given source IDs. Used to power the Clear-button confirm."""
        n = 0
        total = 0
        for sid in source_ids:
            d = get_cache_dir() / "thumbs" / sid
            if not d.exists():
                continue
            for entry in d.iterdir():
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                        n += 1
                    except OSError:
                        pass
        return n, total

    def clear_disk_thumbnails(self, source_ids: Iterable[str]
                              ) -> Tuple[int, int]:
        """Delete the cached-thumbnail directories for the given source IDs
        and clear matching in-memory entries. Returns the (file_count,
        total_bytes) freed. Tolerates missing dirs and partial failures."""
        ids = list(source_ids)
        n, total = self.disk_thumbnail_stats(ids)
        for sid in ids:
            d = get_cache_dir() / "thumbs" / sid
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            # Clear matching mem-cache + lq_emitted markers so the next
            # viewport request actually triggers a fresh decode instead of
            # serving the stale in-memory thumbnail.
            self.invalidate_source(sid)
        return n, total

    @staticmethod
    def _disk_thumb_path(source_id: str, frame_num: int) -> Path:
        return get_cache_dir() / "thumbs" / source_id / f"{frame_num}.jpg"

    @classmethod
    def _load_disk_thumbnail(cls, source_id: str,
                             frame_num: int) -> Optional[QImage]:
        path = cls._disk_thumb_path(source_id, frame_num)
        if not path.exists():
            return None
        qimg = QImage()
        if not qimg.load(str(path), "JPEG"):
            return None
        return qimg

    @classmethod
    def _save_disk_thumbnail(cls, source_id: str, frame_num: int,
                             qimage: QImage) -> bool:
        path = cls._disk_thumb_path(source_id, frame_num)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return qimage.save(str(path), "JPEG", 85)
        except OSError:
            return False


class BulkCacheJob(QObject):
    """Background task that fills the thumbnail disk cache for a fixed list
    of (source_id, frame_num) targets. Each target is decoded by an
    isolated ffmpeg subprocess (NVDEC where available, CPU fallback) so a
    single bad frame can't crash the whole bake. Results are saved to disk,
    populated into the parent cache's memory, and emitted to any visible
    clip referencing the same source frame."""

    progress = Signal(int, int)  # done, total
    finished = Signal()
    cancelled = Signal()

    def __init__(self, cache: ThumbnailCache,
                 targets: List[Tuple[str, int]], parent=None):
        super().__init__(parent)
        self._cache = cache
        self._targets = targets
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    @property
    def total(self) -> int:
        return len(self._targets)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not self._targets:
            self.progress.emit(0, 0)
            self.finished.emit()
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled = True

    def _run(self):
        diag(f"bulk    _run start total={len(self._targets)}")
        # Tell the coordinator to back off for the duration of this bake so
        # we don't end up with both pools opening PyAV containers on the
        # same source. Cleared in `finally`; the wake_event nudge re-arms
        # the coordinator immediately after.
        self._cache._bulk_in_progress.set()
        total = len(self._targets)
        done = 0
        self.progress.emit(0, total)

        # Group targets by source and resolve sources up front. We split
        # later by light/heavy: light sources stay on the fast PyAV sweep
        # path (one decoder context, many frames per call); heavy sources
        # go through per-frame NVDEC subprocesses for crash isolation.
        by_source: Dict[str, List[int]] = {}
        sources_by_id: Dict[str, VideoSource] = {}
        for source_id, frame_num in self._targets:
            source = self._cache._sources.get(source_id)
            if source is None:
                done += 1
                continue
            sources_by_id[source_id] = source
            by_source.setdefault(source_id, []).append(frame_num)
        if done > 0:
            self.progress.emit(done, total)
        if not by_source:
            self.finished.emit()
            return

        # Build futures of two kinds, both dispatched to the same pool:
        # - LIGHT: PyAV sweep on a chunk of nearby frames (fast, scales
        #          via amortized decode + GIL release).
        # - HEAVY: NVDEC subprocess per frame (slower per call but isolated
        #          and reliable on HEVC/HDR/4K).
        # Each future records how many frames it covers so progress and
        # the result loop stay correct regardless of mode.
        sweep_chunks: List[Tuple[VideoSource, str, List[int]]] = []
        heavy_units: List[Tuple[VideoSource, str, int]] = []
        for sid, frames in by_source.items():
            source = sources_by_id[sid]
            if ThumbnailCache._is_heavy_source(source):
                # Probe NVDEC method up front so 6 workers don't race for
                # the same probe. Cheap (one ~50 ms ffmpeg call per source,
                # cached for the cache's lifetime).
                self._cache._get_bulk_decode_method(source)
                for fn in frames:
                    heavy_units.append((source, sid, fn))
            else:
                # PyAV sweep grouping: only chain frames within a typical
                # GOP (SEQUENTIAL_THRESHOLD); cap each chunk so cancel
                # lands within ~one chunk's decode time.
                MAX_RUN = 16
                frames_sorted = sorted(frames)
                raw_runs = ThumbnailCache._build_runs(
                    [(f, "", "") for f in frames_sorted])
                for r in raw_runs:
                    for i in range(0, len(r), MAX_RUN):
                        chunk_frames = [f[0] for f in r[i:i + MAX_RUN]]
                        sweep_chunks.append((source, sid, chunk_frames))

        n_light = sum(len(c) for _, _, c in sweep_chunks)
        n_heavy = len(heavy_units)
        diag(f"bulk    pool create light_workers={_BULK_LIGHT_WORKERS} "
             f"light_chunks={len(sweep_chunks)} (frames={n_light}) "
             f"heavy_workers={_BULK_HEAVY_WORKERS} "
             f"heavy_units={n_heavy}")
        # Two pools: light gets up to cpu_count workers (PyAV scales with
        # CPU), heavy stays capped (NVDEC concurrent-stream limit). Only
        # spin up a pool if we actually have work for it, so a pure-light
        # or pure-heavy bake doesn't carry an unused executor's overhead.
        light_pool = (
            ThreadPoolExecutor(max_workers=_BULK_LIGHT_WORKERS,
                               thread_name_prefix="bulk-light")
            if sweep_chunks else None)
        heavy_pool = (
            ThreadPoolExecutor(max_workers=_BULK_HEAVY_WORKERS,
                               thread_name_prefix="bulk-heavy")
            if heavy_units else None)
        try:
            # futures[fut] = (source_id, frame_count). The count lets
            # progress increment correctly whether the future was a one-
            # frame heavy unit or a multi-frame light chunk.
            # Workers do their own post-processing (save + cache + emit)
            # so the orchestrator stays a thin progress sink. With the old
            # design the orchestrator was the wall-time floor (~3-5 ms/
            # frame of serialized save+emit); now the bulk pool actually
            # scales with workers.
            futures: Dict[Future, int] = {}
            if light_pool is not None:
                for source, sid, chunk_frames in sweep_chunks:
                    if self._cancelled:
                        break
                    fut = light_pool.submit(
                        self._cache._bulk_worker_light,
                        source, sid, chunk_frames)
                    futures[fut] = len(chunk_frames)
            if heavy_pool is not None:
                for source, sid, frame_num in heavy_units:
                    if self._cancelled:
                        break
                    fut = heavy_pool.submit(
                        self._cache._bulk_worker_heavy,
                        source, sid, frame_num)
                    futures[fut] = 1
            diag(f"bulk    submitted {len(futures)} futures, draining…")

            for fut in as_completed(futures):
                if self._cancelled:
                    diag("bulk    cancel detected mid-drain, breaking")
                    break
                n = futures[fut]
                try:
                    fut.result()  # propagate worker exception, ignore value
                except Exception as e:
                    diag(f"bulk    worker EXC {type(e).__name__}: {e}")
                    logger.debug("bulk worker failed: %s", e)
                done += n
                self.progress.emit(done, total)
            diag(f"bulk    drain done={done}/{total}")
        finally:
            # cancel_futures cancels pending-but-not-running futures. In-
            # flight subprocesses aren't interruptible from here; cancel()
            # on the dialog already issued kills on registered procs.
            diag("bulk    pool shutdown begin (wait=True)")
            if light_pool is not None:
                light_pool.shutdown(wait=True, cancel_futures=True)
            if heavy_pool is not None:
                heavy_pool.shutdown(wait=True, cancel_futures=True)
            diag("bulk    pool shutdown done")
            # Re-arm the coordinator: clear the quiesce flag and nudge its
            # wake event so it picks up viewport work immediately.
            self._cache._bulk_in_progress.clear()
            self._cache._wake_event.set()

        if self._cancelled:
            diag("bulk    emit cancelled")
            self.cancelled.emit()
        else:
            diag("bulk    emit finished")
            self.finished.emit()
