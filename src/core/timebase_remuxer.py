"""Remux unsafe-timebase video files in-place via ffmpeg ``-c copy``.

Driven by the timebase warning dialog: when the user clicks Auto-fix, a
``RemuxJob`` runs through the supplied list sequentially, emitting per-source
progress and per-source success so the UI can relink each file as soon as it
finishes (a partial cancel still leaves successfully-remuxed sources fixed).

Pure stream copy — no re-encode, no quality loss. The only thing that
changes is the container's video time_base, set via
``-video_track_timescale <N>`` to a value that divides the source's frame
duration cleanly. See ``core.video_source._frame_duration_for_fps`` for
how the target tick rate is picked per source fps.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Tuple

from PySide6.QtCore import QThread, Signal


logger = logging.getLogger(__name__)


# Job tuple shape: (source_id, input_path, output_path, target_timescale)
RemuxJobSpec = Tuple[str, str, str, int]


class RemuxJob(QThread):
    """Sequential ffmpeg ``-c copy`` runner with cancel + per-source signals.

    Emits one of:
      - ``source_done(source_id, output_path)`` per successful remux
      - ``error(source_id, message)`` per failed remux (non-fatal, batch
        continues unless the caller decides otherwise)
      - ``cancelled()`` once if the user hits cancel mid-batch
      - ``finished_all(succeeded, total)`` once at the end (also fires after
        a cancel — caller checks ``succeeded < total`` to know whether all
        jobs ran)
    """

    progress = Signal(int, int, str)        # (done, total, current_basename)
    source_done = Signal(str, str)          # (source_id, fixed_path)
    error = Signal(str, str)                # (source_id, message)
    cancelled = Signal()
    finished_all = Signal(int, int)         # (succeeded, total)

    def __init__(self, jobs: List[RemuxJobSpec], parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._cancelled = False
        # Single ffmpeg subprocess at a time — sequential by design so cancel
        # only has to kill one. List for parity with scene_detector's pattern.
        self._procs: list = []

    def cancel(self):
        self._cancelled = True
        for proc in list(self._procs):
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        total = len(self._jobs)
        succeeded = 0
        for idx, (source_id, in_path, out_path, target_ts) in enumerate(
                self._jobs):
            if self._cancelled:
                break
            self.progress.emit(idx, total, os.path.basename(in_path))

            cmd = [
                "ffmpeg", "-y", "-nostdin", "-v", "error",
                "-i", in_path,
                "-c", "copy",
                "-video_track_timescale", str(target_ts),
                out_path,
            ]
            logger.info("Remuxing %s → %s (timescale=%d)",
                        in_path, out_path, target_ts)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._procs.append(proc)
            try:
                _, stderr_data = proc.communicate()
            finally:
                if proc in self._procs:
                    self._procs.remove(proc)

            if self._cancelled:
                # ffmpeg killed mid-write leaves a partial file ffprobe
                # can't read — clean it up so the user doesn't think the
                # fix succeeded.
                self._cleanup_partial(out_path)
                break

            if proc.returncode != 0:
                err = (stderr_data or b"").decode("utf-8", errors="replace")
                err = err.strip() or f"ffmpeg exited {proc.returncode}"
                self._cleanup_partial(out_path)
                self.error.emit(source_id, err)
                continue

            succeeded += 1
            self.source_done.emit(source_id, out_path)

        if self._cancelled:
            self.cancelled.emit()
        # Always emit the terminal signal so the dialog's Close-button
        # transition runs even after a cancel — caller compares
        # succeeded vs total to render the right summary.
        self.progress.emit(total, total, "")
        self.finished_all.emit(succeeded, total)

    @staticmethod
    def _cleanup_partial(path: str):
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning("Could not remove partial remux output: %s", path)
