"""System diagnostics for the Tools → System Performance Check dialog.

Runs a set of static capability probes covering every GPU / codec / filter
piece of the pipeline that can silently regress (driver update, ffmpeg
swap, OmniShotCut sidecar deletion, etc.) and reports each as a
``CheckResult`` with a suggested fix when something isn't right.

All probes are best-effort: any unexpected exception inside a probe becomes
a FAIL row with the exception text as detail — the dialog must never crash
mid-check.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_FFMPEG_TIMEOUT = 10  # seconds for any single ffmpeg subprocess


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str         # PASS / WARN / FAIL / SKIP
    detail: str
    fix: Optional[str] = None


def _libmpv_path() -> Path:
    # diagnostics.py lives at src/utils/diagnostics.py → src/ is parents[1].
    return Path(__file__).resolve().parents[1] / "libmpv-2.dll"


def _run_ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", *args],
        capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Individual probes


def check_ffmpeg() -> CheckResult:
    path = shutil.which("ffmpeg")
    if path is None:
        return CheckResult(
            "ffmpeg on PATH", FAIL, "ffmpeg not found on PATH.",
            "Install via `winget install Gyan.FFmpeg` and reopen this "
            "dialog from a fresh shell so PATH updates.",
        )
    try:
        result = _run_ffmpeg(["-version"])
    except (subprocess.TimeoutExpired, OSError) as e:
        return CheckResult(
            "ffmpeg on PATH", FAIL,
            f"ffmpeg found at {path} but `-version` failed: {e}",
            "Reinstall ffmpeg via `winget install Gyan.FFmpeg`.",
        )
    first_line = (result.stdout.splitlines() or [""])[0]
    return CheckResult("ffmpeg on PATH", PASS, first_line)


def check_ffprobe() -> CheckResult:
    path = shutil.which("ffprobe")
    if path is None:
        return CheckResult(
            "ffprobe on PATH", FAIL, "ffprobe not found on PATH.",
            "Install via `winget install Gyan.FFmpeg` (ships ffprobe "
            "alongside ffmpeg) and reopen from a fresh shell.",
        )
    try:
        result = subprocess.run(
            ["ffprobe", "-hide_banner", "-version"],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return CheckResult(
            "ffprobe on PATH", FAIL,
            f"ffprobe found at {path} but `-version` failed: {e}",
            "Reinstall ffmpeg via `winget install Gyan.FFmpeg`.",
        )
    first_line = (result.stdout.splitlines() or [""])[0]
    return CheckResult("ffprobe on PATH", PASS, first_line)


def check_libmpv() -> CheckResult:
    path = _libmpv_path()
    if not path.exists():
        return CheckResult(
            "libmpv-2.dll in src/", FAIL,
            f"Missing: {path}",
            "Download an mpv build from mpv.io and copy `libmpv-2.dll` "
            "into the `src/` directory.",
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    return CheckResult(
        "libmpv-2.dll in src/", PASS, f"{path} ({size_mb:.1f} MB)",
    )


def check_torch_cuda() -> CheckResult:
    try:
        import torch
    except ImportError as e:
        return CheckResult(
            "torch + CUDA", FAIL, f"torch import failed: {e}",
            "Reinstall: `venv\\Scripts\\pip install torch "
            "--index-url https://download.pytorch.org/whl/cu126`",
        )
    if not torch.cuda.is_available():
        return CheckResult(
            "torch + CUDA", FAIL,
            f"torch {torch.__version__} installed but CUDA not available "
            "(likely the CPU-only wheel).",
            "Reinstall the CUDA build: `venv\\Scripts\\pip install "
            "--force-reinstall torch --index-url "
            "https://download.pytorch.org/whl/cu126`",
        )
    try:
        name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        cuda_ver = torch.version.cuda
    except Exception as e:
        return CheckResult(
            "torch + CUDA", WARN,
            f"torch.cuda available but device query failed: {e}",
            None,
        )
    return CheckResult(
        "torch + CUDA", PASS,
        f"torch {torch.__version__} / CUDA {cuda_ver} / {name} ({vram_mb} MB)",
    )


def check_nvdec_capability() -> CheckResult:
    """Confirm ffmpeg was *built* with CUDA hardware decode support."""
    try:
        hwaccels = _run_ffmpeg(["-hwaccels"]).stdout
        decoders = _run_ffmpeg(["-decoders"]).stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        return CheckResult(
            "NVDEC (build support)", FAIL, f"ffmpeg probe failed: {e}",
            "Install via `winget install Gyan.FFmpeg`.",
        )
    has_cuda = "cuda" in hwaccels
    has_h264 = "h264_cuvid" in decoders
    has_hevc = "hevc_cuvid" in decoders
    if has_cuda and has_h264 and has_hevc:
        return CheckResult(
            "NVDEC (build support)", PASS,
            "ffmpeg has cuda hwaccel + h264_cuvid + hevc_cuvid decoders.",
        )
    missing = []
    if not has_cuda:
        missing.append("cuda hwaccel")
    if not has_h264:
        missing.append("h264_cuvid")
    if not has_hevc:
        missing.append("hevc_cuvid")
    return CheckResult(
        "NVDEC (build support)", WARN,
        "ffmpeg missing: " + ", ".join(missing) + ".",
        "Use Gyan.FFmpeg's `full` build (the winget package is full by "
        "default — if you swapped to `essentials`, switch back).",
    )


def check_nvdec_runtime_from_value(current: Optional[str]) -> CheckResult:
    """Build a NVDEC-runtime row from a precomputed mpv hwdec-current value.

    Splitting this out lets the dialog read `hwdec_current` on the UI
    thread (mpv property access isn't thread-safe) and then hand the
    plain string into a background worker.

    `current=None` means "no clip loaded yet". The "no preview widget at
    all" and "exception reading mpv" cases are handled by callers since
    they need direct widget access.
    """
    if current is None:
        return CheckResult(
            "NVDEC (runtime engaged)", SKIP,
            "No clip loaded in the preview. Load a video clip and click "
            "Refresh to verify NVDEC engages.",
        )
    if current in ("no", "", "auto"):
        # 'auto' before mpv has resolved an actual decoder, 'no' after it
        # explicitly fell back to software.
        return CheckResult(
            "NVDEC (runtime engaged)", WARN,
            f"mpv hwdec-current = '{current}' — playback is on the CPU "
            "decode path.",
            "Update the NVIDIA driver or check that the source codec is "
            "supported by NVDEC on this GPU.",
        )
    return CheckResult(
        "NVDEC (runtime engaged)", PASS,
        f"mpv hwdec-current = '{current}' — GPU decode is engaged.",
    )


def check_nvdec_runtime(preview_widget=None) -> CheckResult:
    """Query mpv's currently-active hwdec to confirm NVDEC actually engaged
    at runtime — the capability check tells us only that ffmpeg was built
    with it, not that the live preview is using it."""
    if preview_widget is None:
        return CheckResult(
            "NVDEC (runtime engaged)", SKIP,
            "No preview widget available — open the main window first.",
        )
    try:
        current = preview_widget.get_hwdec_current()
    except Exception as e:
        return CheckResult(
            "NVDEC (runtime engaged)", FAIL,
            f"Reading mpv hwdec-current raised: {e}",
            None,
        )
    return check_nvdec_runtime_from_value(current)


def check_nvenc() -> CheckResult:
    try:
        encoders = _run_ffmpeg(["-encoders"]).stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        return CheckResult(
            "NVENC encoders", FAIL, f"ffmpeg probe failed: {e}",
            "Install via `winget install Gyan.FFmpeg`.",
        )
    has_h264 = "h264_nvenc" in encoders
    has_hevc = "hevc_nvenc" in encoders
    if has_h264 and has_hevc:
        return CheckResult(
            "NVENC encoders", PASS,
            "h264_nvenc + hevc_nvenc available.",
        )
    missing = []
    if not has_h264:
        missing.append("h264_nvenc")
    if not has_hevc:
        missing.append("hevc_nvenc")
    return CheckResult(
        "NVENC encoders", FAIL,
        "Missing: " + ", ".join(missing) + ".",
        "Switch to Gyan.FFmpeg's `full` build (NVENC is excluded from the "
        "`essentials` build).",
    )


def check_opencl_tonemap() -> CheckResult:
    """Reuse the cached probe from core.exporter so we never re-probe a
    second time."""
    try:
        from core.exporter import _probe_gpu_tonemap
    except ImportError as e:
        return CheckResult(
            "OpenCL tonemap (HDR)", FAIL,
            f"Could not import core.exporter._probe_gpu_tonemap: {e}",
            None,
        )
    available, device = _probe_gpu_tonemap()
    if available:
        return CheckResult(
            "OpenCL tonemap (HDR)", PASS,
            f"tonemap_opencl works on device {device}.",
        )
    return CheckResult(
        "OpenCL tonemap (HDR)", WARN,
        "No working OpenCL device found for tonemap_opencl.",
        "Update the GPU driver — OpenCL runtime is required for HDR "
        "exports. SDR exports are unaffected.",
    )


def check_scale_cuda() -> CheckResult:
    try:
        filters = _run_ffmpeg(["-filters"]).stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        return CheckResult(
            "scale_cuda filter", FAIL, f"ffmpeg probe failed: {e}",
            "Install via `winget install Gyan.FFmpeg`.",
        )
    if "scale_cuda" in filters:
        return CheckResult(
            "scale_cuda filter", PASS,
            "Available — zero-copy NVDEC -> scale_cuda -> NVENC path active.",
        )
    return CheckResult(
        "scale_cuda filter", WARN,
        "scale_cuda not compiled into this ffmpeg build.",
        "Use Gyan.FFmpeg's `full` build. Exports still work but fall back "
        "to slower CPU scaling.",
    )


def check_omnishotcut() -> CheckResult:
    try:
        from core.omnishotcut_runner import is_setup_complete
    except ImportError as e:
        return CheckResult(
            "OmniShotCut sidecar (optional)", WARN,
            f"Could not import omnishotcut_runner: {e}", None,
        )
    if is_setup_complete():
        return CheckResult(
            "OmniShotCut sidecar (optional)", PASS,
            "Sentinel + venv + repo all present.",
        )
    return CheckResult(
        "OmniShotCut sidecar (optional)", WARN,
        "Optional transformer scene detector is not installed.",
        "Run `venv\\Scripts\\python.exe scripts\\setup_omnishotcut.py`. "
        "TransNetV2 keeps working without it.",
    )


# ---------------------------------------------------------------------------
# Top-level orchestration


def _safe(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Wrap a probe so any unhandled exception becomes a FAIL row instead
    of crashing the dialog."""
    try:
        return fn()
    except Exception as e:
        logger.exception("Diagnostics probe %s raised", name)
        return CheckResult(name, FAIL, f"Probe crashed: {e!r}", None)


ProbeFn = Callable[[], CheckResult]


# Probe display order. The dialog iterates this registry one probe at a
# time off the UI thread so each row can show progress; scripts/system_check
# uses the legacy synchronous run_all_checks below.
#
# The NVDEC-runtime entry calls check_nvdec_runtime(None) in this registry,
# which yields the "no preview widget" SKIP — the dialog overrides this
# entry by precomputing the row from the live preview widget before kicking
# off the worker (mpv property access must stay on the UI thread).
_PROBE_REGISTRY: List[tuple] = [
    ("ffmpeg on PATH",                 check_ffmpeg),
    ("ffprobe on PATH",                check_ffprobe),
    ("libmpv-2.dll in src/",           check_libmpv),
    ("torch + CUDA",                   check_torch_cuda),
    ("NVDEC (build support)",          check_nvdec_capability),
    ("NVDEC (runtime engaged)",        lambda: check_nvdec_runtime(None)),
    ("NVENC encoders",                 check_nvenc),
    ("OpenCL tonemap (HDR)",           check_opencl_tonemap),
    ("scale_cuda filter",              check_scale_cuda),
    ("OmniShotCut sidecar (optional)", check_omnishotcut),
]


def probe_names() -> List[str]:
    return [name for name, _ in _PROBE_REGISTRY]


def run_probe(index: int) -> CheckResult:
    """Run a single probe from the registry. Used by the async dialog."""
    name, fn = _PROBE_REGISTRY[index]
    return _safe(name, fn)


def run_all_checks(preview_widget=None) -> List[CheckResult]:
    """Run every probe in display order. ``preview_widget`` is forwarded
    to the runtime-mpv probe and ignored by everything else."""
    return [
        _safe("ffmpeg on PATH",                check_ffmpeg),
        _safe("ffprobe on PATH",               check_ffprobe),
        _safe("libmpv-2.dll in src/",          check_libmpv),
        _safe("torch + CUDA",                  check_torch_cuda),
        _safe("NVDEC (build support)",         check_nvdec_capability),
        _safe("NVDEC (runtime engaged)",       lambda: check_nvdec_runtime(preview_widget)),
        _safe("NVENC encoders",                check_nvenc),
        _safe("OpenCL tonemap (HDR)",          check_opencl_tonemap),
        _safe("scale_cuda filter",             check_scale_cuda),
        _safe("OmniShotCut sidecar (optional)", check_omnishotcut),
    ]


def format_as_text(results: List[CheckResult]) -> str:
    """Plain-text rendering for the Copy-to-clipboard button."""
    lines = []
    for r in results:
        lines.append(f"[{r.status:4s}] {r.name} — {r.detail}")
        if r.fix:
            lines.append(f"        Fix: {r.fix}")
    return "\n".join(lines)
