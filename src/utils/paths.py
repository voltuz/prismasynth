import os
from pathlib import Path

APP_NAME = "prismasynth"


def get_cache_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    d = base / APP_NAME / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_config_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
