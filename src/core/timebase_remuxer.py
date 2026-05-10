"""Remux unsafe-timebase video files via ffmpeg.

Driven by the timebase warning dialog: when the user clicks Auto-fix, a
``RemuxJob`` runs through the supplied list sequentially, emitting per-source
progress and per-source success so the UI can relink each file as soon as it
finishes (a partial cancel still leaves successfully-remuxed sources fixed).

Video is always stream-copied — no re-encode, no quality loss. The only
thing that changes on the video side is the container's video time_base,
set via ``-video_track_timescale <N>`` to a value that divides the source's
frame duration cleanly. See ``core.video_source._frame_duration_for_fps``
for how the target tick rate is picked per source fps.

Audio handling is per-run, controlled by the ``audio_mode`` constructor arg:

  - ``AUDIO_KEEP``           — ``-c:a copy`` (preserves original codec + layout)
  - ``AUDIO_REENCODE_SAME``  — convert to 16-bit PCM at 48 kHz, original layout
  - ``AUDIO_STEREO``         — convert to 16-bit PCM at 48 kHz stereo
                               (surround sources are downmixed via ffmpeg's
                               standard matrix; no audio is lost — channels
                               are condensed into the stereo pair)

Per-source progress is parsed from ffmpeg's ``-progress pipe:1`` stdout.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import List, Tuple

from PySide6.QtCore import QThread, Signal


logger = logging.getLogger(__name__)


# Job tuple shape: (source_id, input_path, output_path, target_timescale,
#                   duration_seconds)
RemuxJobSpec = Tuple[str, str, str, int, float]


class RemuxJob(QThread):
    """Sequential ffmpeg runner with cancel + per-source signals.

    Emits one of:
      - ``source_done(source_id, output_path)`` per successful remux
      - ``error(source_id, message)`` per failed remux (non-fatal, batch
        continues unless the caller decides otherwise)
      - ``cancelled()`` once if the user hits cancel mid-batch
      - ``finished_all(succeeded, total)`` once at the end (also fires after
        a cancel — caller checks ``succeeded < total`` to know whether all
        jobs ran)
    """

    AUDIO_KEEP = "keep"
    AUDIO_REENCODE_SAME = "reencode_same"
    AUDIO_STEREO = "stereo"

    progress = Signal(int, int, str)        # (done, total, current_basename)
    source_progress = Signal(str, float)    # (source_id, fraction 0.0..1.0)
    source_done = Signal(str, str)          # (source_id, fixed_path)
    error = Signal(str, str)                # (source_id, message)
    cancelled = Signal()
    finished_all = Signal(int, int)         # (succeeded, total)

    def __init__(self, jobs: List[RemuxJobSpec],
                 audio_mode: str = AUDIO_KEEP, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._audio_mode = audio_mode
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

    # ------------------------------------------------------------------

    def _build_cmd(self, in_path: str, out_path: str,
                   target_ts: int) -> list:
        """Compose the ffmpeg command for one job. Video is always
        stream-copied; only the audio tail varies by audio_mode.

        ``-map 0:v:0 -map 0:a:0?`` selects the first video and (if present)
        the first audio stream — matches what ``core.exporter`` does for
        export pipelines and discards Bluray-rip extras like commentary
        tracks. The trailing ``?`` makes the audio map optional, so silent
        sources still remux successfully.
        """
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-v", "error",
            "-progress", "pipe:1",
            "-i", in_path,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "copy",
            "-video_track_timescale", str(target_ts),
        ]
        if self._audio_mode == self.AUDIO_KEEP:
            cmd += ["-c:a", "copy"]
        elif self._audio_mode == self.AUDIO_REENCODE_SAME:
            cmd += ["-c:a", "pcm_s16le", "-ar", "48000"]
        elif self._audio_mode == self.AUDIO_STEREO:
            # ``-ac 2`` invokes ffmpeg's default downmix matrix which
            # condenses surround channels into L/R rather than dropping
            # them, matching the user's "no audio is lost" requirement.
            cmd += ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]
        cmd += [out_path]
        return cmd

    # ------------------------------------------------------------------

    def run(self):
        total = len(self._jobs)
        succeeded = 0
        for idx, job in enumerate(self._jobs):
            if self._cancelled:
                break
            source_id, in_path, out_path, target_ts, duration_s = job
            self.progress.emit(idx, total, os.path.basename(in_path))
            self.source_progress.emit(source_id, 0.0)

            cmd = self._build_cmd(in_path, out_path, target_ts)
            logger.info("Remuxing %s -> %s (timescale=%d, audio=%s)",
                        in_path, out_path, target_ts, self._audio_mode)

            stderr_data = self._run_one(cmd, source_id, duration_s)

            if self._cancelled:
                # ffmpeg killed mid-write leaves a partial file ffprobe
                # can't read — clean it up so the user doesn't think the
                # fix succeeded.
                self._cleanup_partial(out_path)
                break

            if stderr_data is None:
                # _run_one signalled an internal error (already logged).
                self._cleanup_partial(out_path)
                continue

            proc_returncode = stderr_data[0]
            stderr_bytes = stderr_data[1]

            if proc_returncode != 0:
                err = stderr_bytes.decode("utf-8", errors="replace").strip()
                err = err or f"ffmpeg exited {proc_returncode}"
                self._cleanup_partial(out_path)
                self.error.emit(source_id, err)
                continue

            self.source_progress.emit(source_id, 1.0)
            succeeded += 1
            self.source_done.emit(source_id, out_path)

        if self._cancelled:
            self.cancelled.emit()
        # Always emit the terminal signal so the dialog's Close-button
        # transition runs even after a cancel — caller compares
        # succeeded vs total to render the right summary.
        self.progress.emit(total, total, "")
        self.finished_all.emit(succeeded, total)

    def _run_one(self, cmd: list, source_id: str,
                 duration_s: float):
        """Run one ffmpeg invocation, parsing -progress lines from stdout
        in the foreground and draining stderr in a daemon thread (avoids
        the OS-pipe-fills-and-deadlocks failure mode documented in
        ``core.exporter._drain``).

        Returns (returncode, stderr_bytes) on completion, or None when an
        unexpected internal exception aborts the run.
        """
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.error.emit(source_id, f"Failed to launch ffmpeg: {e}")
            return None
        self._procs.append(proc)

        stderr_chunks: list = []

        def _drain_stderr():
            try:
                for chunk in iter(proc.stderr.readline, b""):
                    stderr_chunks.append(chunk)
            except Exception:
                pass

        drain = threading.Thread(target=_drain_stderr, daemon=True)
        drain.start()

        last_pct_emitted = -1.0
        try:
            for raw in iter(proc.stdout.readline, b""):
                if self._cancelled:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("out_time_us="):
                    continue
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if duration_s <= 0:
                    continue
                pct = us / 1_000_000.0 / duration_s
                if pct < 0:
                    pct = 0.0
                elif pct > 1.0:
                    pct = 1.0
                # Throttle: emit only when advancing by >=1% to keep the
                # Qt event queue from drowning in tiny updates on long
                # remuxes. The terminal 1.0 emit lives in run().
                if pct - last_pct_emitted >= 0.01:
                    self.source_progress.emit(source_id, pct)
                    last_pct_emitted = pct
        finally:
            try:
                proc.wait()
            except Exception:
                pass
            drain.join(timeout=2.0)
            if proc in self._procs:
                self._procs.remove(proc)

        return proc.returncode, b"".join(stderr_chunks)

    @staticmethod
    def _cleanup_partial(path: str):
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning("Could not remove partial remux output: %s", path)
