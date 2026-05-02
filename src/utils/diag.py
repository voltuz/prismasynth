"""Lightweight diagnostic logging — append-only, flush-on-write so the log
survives crashes / hangs / hard kills. Intended for short-term troubleshooting;
remove the call sites once an issue is solved.

Two entry points:
    diag(msg) — timestamp + thread-name + msg, one line.
    dump_all_thread_stacks() — every Python thread's stack, for hang triage.

Logs go next to crash.log so users can find them in one place.
"""
import os
import sys
import threading
import time
import traceback
from datetime import datetime

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # src/
    "diag.log",
)
_lock = threading.Lock()
_t0 = time.monotonic()


def _write(text: str):
    try:
        with _lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass
    except OSError:
        pass


def diag(msg: str):
    """Log a single timestamped breadcrumb. Cheap, ~5 µs per call."""
    elapsed = time.monotonic() - _t0
    name = threading.current_thread().name
    _write(f"{elapsed:8.3f} [{name:>20}] {msg}\n")


def dump_all_thread_stacks(reason: str = "manual dump"):
    """Walk every live Python thread, format its current stack, append to
    the diag log. Use during a hang to see where each thread is blocked."""
    parts = [
        f"\n=========================================================\n"
        f"=== {datetime.now().isoformat()}  THREAD DUMP: {reason}\n"
        f"=========================================================\n"
    ]
    threads = {t.ident: t for t in threading.enumerate()}
    frames = sys._current_frames()
    for tid, frame in frames.items():
        t = threads.get(tid)
        name = t.name if t else f"?{tid}"
        daemon = "daemon" if (t and t.daemon) else "main "
        parts.append(f"\n--- Thread {name!r} ({daemon}, id={tid}) ---\n")
        parts.extend(traceback.format_stack(frame))
    parts.append(f"\n=== end of dump ({len(frames)} threads) ===\n\n")
    _write("".join(parts))
