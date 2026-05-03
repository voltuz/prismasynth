"""Main-process wrapper for the OmniShotCut sidecar subprocess.

Spawns scripts/omnishotcut_sidecar.py inside venv-omnishotcut/, decodes each
segment's frames via the shared ffmpeg_decode pipeline at the model's required
resolution, streams them to the sidecar over stdin, and dispatches per-segment
results / progress back to the caller via callbacks.

The sidecar is launched once per Detect Cuts run and reused for all segments —
the model load (5-15s) is amortised, not repeated per segment.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from core.video_source import VideoSource
from core.ffmpeg_decode import decode_to_array

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    # src/core/omnishotcut_runner.py → repo root
    return Path(__file__).resolve().parents[2]


def venv_python_path() -> Path:
    """Path to the sidecar venv's Python interpreter."""
    return _project_root() / "venv-omnishotcut" / "Scripts" / "python.exe"


def setup_sentinel_path() -> Path:
    """Path to the file that, when present, indicates setup completed successfully."""
    return _project_root() / "venv-omnishotcut" / ".prismasynth_ready"


def omnishotcut_repo_path() -> Path:
    """Path to the vendored OmniShotCut clone."""
    return _project_root() / "third_party" / "OmniShotCut"


def sidecar_script_path() -> Path:
    return _project_root() / "scripts" / "omnishotcut_sidecar.py"


def is_setup_complete() -> bool:
    """True when the sidecar venv + checkpoint setup has finished successfully.
    Checks the sentinel file, NOT just python.exe — the venv may exist mid-setup
    before deps are installed, and we must not launch a sidecar in that state."""
    return (setup_sentinel_path().exists()
            and venv_python_path().exists()
            and omnishotcut_repo_path().exists())


def default_checkpoint_path() -> Path:
    """Where the OmniShotCut .pth file is downloaded to."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home())
    return Path(base) / "prismasynth" / "models" / "OmniShotCut_ckpt.pth"


class OmnishotcutRunner:
    """Drives a single end-to-end OmniShotCut detection run via the sidecar.

    Constructor callbacks (all optional except the segment-done one):
      on_phase(phase: str)
      on_decode_progress(seg_done: int)
      on_analyze_progress(frame_done: int, frame_total: int, seg_id)
      on_segment_done(clip_id, source: VideoSource, range_start: int,
                      seg_total: int, cuts: list[int])
    """

    def __init__(self, segments: list, sources: dict, checkpoint_path: str,
                 procs: list, is_cancelled: Callable[[], bool],
                 on_phase: Optional[Callable] = None,
                 on_decode_progress: Optional[Callable] = None,
                 on_analyze_progress: Optional[Callable] = None,
                 on_segment_done: Optional[Callable] = None,
                 num_context_frames: int = 0):
        self._segments = segments
        self._sources = sources
        self._checkpoint_path = checkpoint_path
        self._procs = procs
        self._is_cancelled = is_cancelled
        self._on_phase = on_phase or (lambda _p: None)
        self._on_decode_progress = on_decode_progress or (lambda _d: None)
        self._on_analyze_progress = on_analyze_progress or (lambda _f, _t, _s: None)
        self._on_segment_done = on_segment_done or (lambda *_a: None)
        self._num_context_frames = num_context_frames

        self._proc: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._loaded_q: "queue.Queue" = queue.Queue()
        self._result_q: "queue.Queue" = queue.Queue()
        self._error_q: "queue.Queue" = queue.Queue()
        self._stderr_buf: list = []

    # --- Lifecycle ---

    def cancel(self):
        """Best-effort graceful cancel: CTRL_BREAK_EVENT first, then kill after 2s."""
        proc = self._proc
        if proc is None:
            return
        try:
            if sys.platform == "win32":
                proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        """Run all segments through the sidecar. Blocks until done or cancelled.
        Raises on fatal errors (sidecar exit, decode failure, etc.)."""
        if not is_setup_complete():
            raise RuntimeError(
                "OmniShotCut is not set up. Run scripts/setup_omnishotcut.py first."
            )
        if not Path(self._checkpoint_path).exists():
            raise FileNotFoundError(
                f"OmniShotCut checkpoint missing: {self._checkpoint_path}"
            )

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._on_phase("Loading model")
        cmd = [str(venv_python_path()), str(sidecar_script_path())]
        logger.info("Spawning OmniShotCut sidecar: %s", cmd)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creationflags,
        )
        self._procs.append(self._proc)

        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        try:
            self._send_header()
            self._wait_for_loaded()  # populates self._process_width/height
            for segment in self._segments:
                if self._is_cancelled():
                    return
                self._process_segment(segment)
        finally:
            self._shutdown()

    # --- Protocol I/O ---

    def _send_header(self):
        header = {
            "omnishotcut_repo": str(omnishotcut_repo_path()),
            "checkpoint": str(self._checkpoint_path),
            "num_context_frames": int(self._num_context_frames),
        }
        self._write_line(header)

    def _wait_for_loaded(self):
        """Block until sidecar emits "loaded" or "error". Stash model dims on self."""
        while True:
            try:
                msg = self._loaded_q.get(timeout=120.0)
            except queue.Empty:
                raise RuntimeError(
                    "OmniShotCut sidecar timed out during model load (120s).\n"
                    + self._stderr_tail()
                )
            if msg.get("phase") == "loaded":
                self._process_width = int(msg["process_width"])
                self._process_height = int(msg["process_height"])
                self._max_window = int(msg["max_window"])
                vram = msg.get("vram_mb", 0)
                logger.info(
                    "OmniShotCut loaded: process=%dx%d, max_window=%d, vram=%dMB",
                    self._process_width, self._process_height, self._max_window, vram,
                )
                return
            if msg.get("phase") == "error":
                raise RuntimeError(f"sidecar error during load: {msg.get('msg')}\n"
                                    + self._stderr_tail())

    def _process_segment(self, segment):
        source_id, source_in, source_out, clip_id = segment
        source = self._sources[source_id]
        seg_total = source_out - source_in + 1
        seg_id = str(clip_id)

        self._on_phase("Decoding")
        padded, n_frames, _, _ = decode_to_array(
            source, source_in, seg_total,
            self._process_width, self._process_height,
            procs=self._procs,
            is_cancelled=self._is_cancelled,
            progress_cb=lambda done, _t: self._on_decode_progress(done),
        )
        if self._is_cancelled():
            return
        if n_frames == 0:
            raise RuntimeError(f"all decode paths failed for {source.file_path}")

        # decode_to_array allocates with no padding when pad_before/after=0,
        # so padded == frames.
        frames = padded[:n_frames]

        self._on_phase("Analyzing")
        seg_msg = {
            "phase": "segment",
            "seg_id": seg_id,
            "frame_count": int(n_frames),
            "fps": float(source.fps if source.fps > 0 else 24.0),
        }
        self._write_line(seg_msg)
        self._write_bytes(frames.tobytes())

        # Drain analyzing progress until result arrives
        while True:
            if self._is_cancelled():
                return
            try:
                msg = self._result_q.get(timeout=300.0)
            except queue.Empty:
                raise RuntimeError(
                    f"OmniShotCut sidecar stalled on segment {seg_id} (300s)\n"
                    + self._stderr_tail()
                )
            if msg.get("phase") == "result" and msg.get("seg_id") == seg_id:
                cuts = [int(c) for c in msg.get("cuts", [])]
                self._on_segment_done(clip_id, source, source_in, seg_total, cuts)
                return
            if msg.get("phase") == "error":
                raise RuntimeError(
                    f"sidecar error on segment {seg_id}: {msg.get('msg')}\n"
                    + self._stderr_tail()
                )

    def _shutdown(self):
        """Close stdin (signals sidecar to exit cleanly), wait, then drain pipes."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logger.warning("OmniShotCut sidecar didn't exit in 10s; killing")
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        finally:
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=2.0)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)
            self._proc = None

    # --- Background pipe drains ---

    def _drain_stdout(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(raw.decode("utf-8"))
            except Exception:
                logger.debug("sidecar stdout (non-JSON): %r", raw)
                continue
            phase = msg.get("phase")
            if phase == "analyzing":
                self._on_analyze_progress(
                    int(msg.get("frame", 0)),
                    int(msg.get("total", 0)),
                    msg.get("seg_id"),
                )
            elif phase == "loaded":
                self._loaded_q.put(msg)
            elif phase == "result":
                self._result_q.put(msg)
            elif phase == "error":
                # Push into both queues so whoever is waiting wakes up
                self._loaded_q.put(msg)
                self._result_q.put(msg)
                self._error_q.put(msg)
            else:
                logger.debug("sidecar stdout: %r", msg)

    def _drain_stderr(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = repr(raw)
            if line:
                logger.info("[sidecar stderr] %s", line)
                self._stderr_buf.append(line)
                # Cap stderr buffer to last 200 lines
                if len(self._stderr_buf) > 200:
                    self._stderr_buf = self._stderr_buf[-200:]

    def _stderr_tail(self) -> str:
        if not self._stderr_buf:
            return ""
        tail = "\n".join(self._stderr_buf[-30:])
        return f"--- sidecar stderr (last 30 lines) ---\n{tail}\n---"

    # --- stdin helpers ---

    def _write_line(self, obj):
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("sidecar stdin not available")
        data = (json.dumps(obj) + "\n").encode("utf-8")
        proc.stdin.write(data)
        proc.stdin.flush()

    def _write_bytes(self, data: bytes):
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("sidecar stdin not available")
        # Write in chunks to avoid pipe-buffer stalls on very large segments
        view = memoryview(data)
        chunk = 1 << 20  # 1 MB
        for i in range(0, len(view), chunk):
            if self._is_cancelled():
                return
            proc.stdin.write(view[i:i + chunk])
        proc.stdin.flush()
