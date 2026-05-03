"""One-shot setup for the OmniShotCut sidecar.

Idempotent. Streams every step to stdout so the in-app setup dialog can tail it.

Steps:
  1. Clone third_party/OmniShotCut/ if missing.
  2. Download .uv/uv.exe (Astral's static Python+pip manager) if missing.
  3. uv python install 3.10
  4. uv venv --python 3.10 venv-omnishotcut
  5. uv pip install torch==2.5.1 torchvision==0.20.1 (CUDA 12.4 wheels)
  6. uv pip install -r third_party/OmniShotCut/requirements.txt
  7. Download OmniShotCut_ckpt.pth to %LOCALAPPDATA%/prismasynth/models/
  8. Run sidecar --selftest (load model, infer on synthetic clip)
  9. Touch venv-omnishotcut/.prismasynth_ready sentinel

Re-running with --repair deletes the sentinel and starts from scratch.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Force UTF-8 on our own stdout/stderr — Windows cmd.exe defaults to cp1252
# which can't encode replacement chars (�) that appear when subprocesses
# write tqdm progress bars or other non-ASCII output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


REPO_ROOT = Path(__file__).resolve().parents[1]
UV_DIR = REPO_ROOT / ".uv"
UV_EXE = UV_DIR / "uv.exe"
VENV_DIR = REPO_ROOT / "venv-omnishotcut"
SENTINEL = VENV_DIR / ".prismasynth_ready"
THIRD_PARTY = REPO_ROOT / "third_party"
OMNISHOTCUT_REPO = THIRD_PARTY / "OmniShotCut"
SIDECAR_SCRIPT = REPO_ROOT / "scripts" / "omnishotcut_sidecar.py"

OMNISHOTCUT_GIT_URL = "https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git"
UV_DOWNLOAD_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-pc-windows-msvc.zip"
)

CHECKPOINT_HF_REPO = "uva-cv-lab/OmniShotCut"
CHECKPOINT_FILENAME = "OmniShotCut_ckpt.pth"


def log(msg: str):
    sys.stdout.write(f"[setup] {msg}\n")
    sys.stdout.flush()


def fail(msg: str, code: int = 1):
    sys.stdout.write(f"[setup] FAIL: {msg}\n")
    sys.stdout.flush()
    sys.exit(code)


def run(cmd, *, env=None, cwd=None):
    """Run a subprocess, streaming both stdout+stderr line-by-line. Raises on nonzero exit."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    # Force UTF-8 in the child too, so its tqdm/print output doesn't emit
    # invalid chars that we'd then have to replace.
    full_env = os.environ.copy() if env is None else dict(env)
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    full_env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=full_env,
        cwd=cwd,
    )
    for line in proc.stdout:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            # Last-resort fallback: drop chars our destination encoding can't handle
            sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii"))
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {cmd[0]}")


def step_clone_omnishotcut():
    if OMNISHOTCUT_REPO.exists() and (OMNISHOTCUT_REPO / "test_code").exists():
        log(f"OmniShotCut repo already present at {OMNISHOTCUT_REPO}")
        return
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    log(f"Cloning OmniShotCut into {OMNISHOTCUT_REPO}")
    run(["git", "clone", "--depth", "1", OMNISHOTCUT_GIT_URL, str(OMNISHOTCUT_REPO)])


def step_install_uv():
    if UV_EXE.exists():
        log(f"uv already at {UV_EXE}")
        return
    UV_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading uv from {UV_DOWNLOAD_URL}")
    try:
        with urllib.request.urlopen(UV_DOWNLOAD_URL, timeout=120) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        fail(f"failed to download uv: {e}\n"
             f"   You can manually install uv from https://docs.astral.sh/uv/getting-started/installation/\n"
             f"   then place uv.exe at {UV_EXE}")
    log(f"Extracting uv ({len(data) // 1024} KB)")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith("uv.exe"):
                with zf.open(name) as src, open(UV_EXE, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
        else:
            fail("uv.exe not found in zip download")
    log(f"uv installed at {UV_EXE}")


def step_install_python_310():
    log("Installing CPython 3.10 via uv")
    run([str(UV_EXE), "python", "install", "3.10"])


def step_create_venv():
    if VENV_DIR.exists():
        log(f"venv already exists at {VENV_DIR}")
        return
    log(f"Creating venv at {VENV_DIR}")
    run([str(UV_EXE), "venv", "--python", "3.10", str(VENV_DIR)])


def venv_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe"


def step_install_torch():
    log("Installing torch 2.5.1 + torchvision 0.20.1 (CUDA 12.4 wheels)")
    run([
        str(UV_EXE), "pip", "install",
        "--python", str(venv_python()),
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])


def step_install_requirements():
    req_file = OMNISHOTCUT_REPO / "requirements.txt"
    if not req_file.exists():
        fail(f"missing {req_file}")
    log(f"Installing OmniShotCut requirements from {req_file}")
    run([
        str(UV_EXE), "pip", "install",
        "--python", str(venv_python()),
        "-r", str(req_file),
    ])


def checkpoint_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home())
    return Path(base) / "prismasynth" / "models" / CHECKPOINT_FILENAME


def step_download_checkpoint():
    target = checkpoint_path()
    if target.exists() and target.stat().st_size > 1024 * 1024:
        log(f"Checkpoint already present at {target} ({target.stat().st_size // (1024 * 1024)} MB)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading OmniShotCut checkpoint to {target}")
    # Use huggingface_hub from inside the sidecar venv (where it's already installed).
    # Atomic download: hf_hub_download writes to a temp .incomplete file then renames.
    download_script = (
        "import os, sys, shutil\n"
        "from huggingface_hub import hf_hub_download\n"
        f"path = hf_hub_download(repo_id={CHECKPOINT_HF_REPO!r}, filename={CHECKPOINT_FILENAME!r})\n"
        f"shutil.copyfile(path, {str(target)!r})\n"
        "print(f'downloaded {os.path.getsize(path) // (1024*1024)} MB')\n"
    )
    run([str(venv_python()), "-c", download_script])
    if not target.exists() or target.stat().st_size < 1024 * 1024:
        fail(f"checkpoint download failed (file missing or too small): {target}")
    log(f"Checkpoint OK: {target} ({target.stat().st_size // (1024 * 1024)} MB)")


def step_selftest():
    log("Running sidecar --selftest")
    run([
        str(venv_python()), str(SIDECAR_SCRIPT),
        "--selftest",
        "--omnishotcut-repo", str(OMNISHOTCUT_REPO),
        "--checkpoint", str(checkpoint_path()),
    ])


def step_write_sentinel():
    SENTINEL.write_text(f"prismasynth setup completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                        encoding="utf-8")
    log(f"Sentinel written: {SENTINEL}")


def main():
    parser = argparse.ArgumentParser(description="Set up OmniShotCut sidecar venv + checkpoint.")
    parser.add_argument("--repair", action="store_true",
                        help="Delete the sentinel file and re-run all steps.")
    args = parser.parse_args()

    if args.repair and SENTINEL.exists():
        log(f"--repair: removing sentinel {SENTINEL}")
        SENTINEL.unlink()

    if SENTINEL.exists() and venv_python().exists() and OMNISHOTCUT_REPO.exists() and checkpoint_path().exists():
        log("Setup already complete (sentinel + venv + repo + checkpoint all present).")
        log("Use --repair to re-run.")
        return 0

    t0 = time.monotonic()
    try:
        step_clone_omnishotcut()
        step_install_uv()
        step_install_python_310()
        step_create_venv()
        step_install_torch()
        step_install_requirements()
        step_download_checkpoint()
        step_selftest()
        step_write_sentinel()
    except Exception as e:
        fail(str(e))
    elapsed = time.monotonic() - t0
    log(f"DONE in {int(elapsed)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
