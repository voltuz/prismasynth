# PrismaSynth

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.8+-green.svg)
![Version](https://img.shields.io/badge/version-v0.15.1-green.svg)
![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)

PrismaSynth is a PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with a GPU neural network, review/trim/split clips on a timeline, tag clips with People (groups) for keyboard-fast curation, and export selected groups as video, audio, image sequence, FCPXML, or OpenTimelineIO. Single-track editor — no layers, no compositing.

Built for speed: the entire preview pipeline stays on GPU (NVDEC decode → GPU buffer → mpv display), and export runs parallel FFmpeg segments with zero-copy NVENC for SDR sources.

## Features

- **GPU scrubbing** — libmpv-backed preview at full quality, NVDEC → GPU buffer → display, never falls back to proxies.
- **Scene detection** — TransNetV2 (GPU, default) or OmniShotCut transformer (sidecar venv, hard-cuts mode); HSV-differencing fallback when CUDA isn't available.
- **People (group) tagging** — colored tags with optional digit shortcuts (0–9). Press a digit to toggle the group on selected clips. Multi-clip behaviour: tags all clips if any are missing the group, untags all if every clip already had it.
- **Project Versions** — durable `.psynth` snapshots in a sibling `*.versions/` folder. Captured on every autosave, on demand, and before risky operations (Detect Cuts, multi-delete, group delete, source removal, restore). Older versions are thinned automatically (1/hour for a day, 1/day for a week, 1/week beyond). Restore is itself reversible — a fresh pre-restore snapshot is always taken first.
- **Frame Snapshot (F12)** — `Tools → Snapshot Frame` writes the current preview frame to a full-resolution PNG in a per-project `_snapshots/` folder. Reuses the exporter's `(frame − 0.5) / fps` seek margin so the saved frame is exactly the one on screen.
- **Five export formats, one dialog** — Video, Image Sequence, Audio Only, FCPXML, OpenTimelineIO. Shared "Include gaps" and "Use in/out range" toggles; shared People-group filter to export only tagged subsets.
- **Audio export modes** — none, embedded, standalone sidecar, or both. Standalone supports WAV, FLAC, MP3, M4A.
- **Source timebase audit & Auto-fix** — flags sources whose container `time_base` can't exactly represent their fps (the cause of ±1-frame drift on FCPXML/OTIO import in NLEs that time-seek the source). One-click in-app FFmpeg remux fixes them, with audio re-encode options (keep / PCM same channels / PCM stereo) and live progress.
- **Frame-accurate timing** — integer-frame export pipeline (`-frames:v N`); FCPXML/OTIO emit rational fractions with the right NTSC seek nudge so Resolve lands on the intended source frame.
- **HDR-aware video** — auto-detects PQ/HLG + BT.2020 sources and routes through `tonemap_opencl=hable` (GPU) with a `zscale` CPU fallback. SDR sources skip tonemap entirely.
- **Project portability** — `.psynth` files store both relative and absolute source paths; missing sources surface a folder-based Relink dialog before clearing the previous project's state.
- **Customizable shortcuts** — every binding rebindable via File → Keyboard Shortcuts. Default WASD layout for the most-used edits.

## Keyboard Shortcuts

WASD layout (left hand in gaming position):

| Key   | Action                  | Key       | Action                    |
|-------|-------------------------|-----------|---------------------------|
| W     | Delete (leave gap)      | E         | Set In point              |
| A     | Select to gap           | R         | Set Out point             |
| S     | Split at playhead       | X         | Clear In/Out              |
| D     | Ripple delete           | C         | Cut mode                  |
| V     | Selection mode          | Space     | Play/Pause                |
| 0–9   | Toggle People group     | Ctrl+D    | Detect Cuts               |

All shortcuts are rebindable via File → Keyboard Shortcuts.

Right-click while dragging the playhead performs a quick-cut at the playhead position without switching modes.

## Requirements

- Python 3.11+
- Windows
- FFmpeg and ffprobe on `PATH`
- `libmpv-2.dll` in `src/` — grab it from an [mpv.io](https://mpv.io/installation/) dev build
- NVIDIA GPU with CUDA 12.6+ for scene detection and NVENC export (optional — CPU fallback paths exist)
- Optional: OmniShotCut alternate detector — see Setup

## Setup

```bash
git clone https://github.com/voltuz/prismasynth.git
cd prismasynth

python -m venv venv
venv\Scripts\pip install -r requirements.txt

# PyTorch with CUDA (for GPU scene detection)
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126

# Optional: OmniShotCut alternate detector (~5 GB download, sidecar venv)
venv\Scripts\python scripts\setup_omnishotcut.py
```

Place `libmpv-2.dll` in `src/`.

## Running

```bash
venv\Scripts\python src\main.py
```

Or double-click `run.bat`.

## Workflow

1. **Import** a video (File → Import, drag-and-drop, or Ctrl+I) — creates one whole-file clip per source.
2. **Detect Cuts** (Ctrl+D) — replaces each clip with one clip per shot. Pick TransNetV2 (default) or OmniShotCut from the dialog.
3. **Review** — scrub with the playhead, delete bad shots with `W`, ripple-delete with `D`, split with `S`, switch to cut mode with `C`.
4. **Tag** clips with People — press a digit (1–9, then 0) with clips selected to bind/toggle a group. The People panel manages colours, digits, and clip counts.
5. **Export** — Timeline → Export… opens the unified dialog (Video / Image / Audio / XML / OTIO). Filter to People groups, include or skip gaps, constrain to the in/out range.

## Architecture

Three-layer design:

```
src/ui/     PySide6 widgets (timeline, preview, dialogs)
src/core/   Business logic, threading, export pipeline, scene detection
src/utils/  ffprobe wrappers, path helpers
```

Entry point is `src/main.py` → `MainWindow` in `src/ui/main_window.py`, which wires everything together.

### Data Model

- `VideoSource` — immutable metadata (path, fps, dimensions, codec, time_base) for an imported file.
- `Clip` — references a source by `source_id` + in/out frame numbers. `source_id=None` means it's a gap. Carries `group_ids` for People tags.
- `Group` — a project-scoped People tag (name, colour, optional digit shortcut).
- `TimelineModel` — ordered list of clips/gaps with selection, undo/redo, and in/out render points. Owns the groups dict. Emits Qt signals on mutation.

All position math uses integer frame numbers. Cumulative frame counts are converted to pixels via `_frame_to_pixel()` / `_pixel_to_frame()` to prevent truncation drift.

### Threading

| Thread                    | Work                                                  | Constraint                            |
|---------------------------|-------------------------------------------------------|---------------------------------------|
| Main (Qt)                 | UI, signal slots, mpv commands                        | QPixmap main-thread only              |
| Thumbnail coordinator     | Persistent thread + 6-worker pool, sweep decode       | Pauses during middle-mouse pan        |
| Scene detection (QThread) | Parallel FFmpeg decode + TransNetV2 inference         | Subprocess refs tracked for cancel    |
| OmniShotCut sidecar       | Separate Python 3.10 process; loads model once         | JSON-line stdio + raw bytes           |
| Export                    | Per-segment FFmpeg via ThreadPoolExecutor             | Signals via `QueuedConnection`        |

mpv and thumbnail generation are paused during scene detection to avoid GPU contention.

## Project Layout

```
src/
├── main.py                   # entry point, dark theme, crash hook
├── core/
│   ├── timeline.py           # clips, gaps, groups, undo/redo
│   ├── clip.py               # Clip + Gap dataclasses
│   ├── group.py              # People tag registry + filter
│   ├── video_source.py       # source metadata + is_seek_safe
│   ├── shortcuts.py          # rebindable key registry
│   ├── exporter.py           # parallel FFmpeg pipeline (Video/Image/Audio)
│   ├── xml_exporter.py       # FCPXML 1.9 writer
│   ├── otio_exporter.py      # OpenTimelineIO JSON writer
│   ├── scene_detector.py     # TransNetV2 + HSV fallback
│   ├── omnishotcut_runner.py # OmniShotCut sidecar driver
│   ├── thumbnail_cache.py    # viewport-prioritized generation
│   ├── timebase_remuxer.py   # FFmpeg remux worker (Auto-fix)
│   ├── ui_scale.py           # percent UI scale
│   └── project.py            # .psynth save/load
├── ui/                       # PySide6 widgets — main_window, timeline,
│                             #   preview, dialogs (export, relink, etc.)
└── utils/
    └── ffprobe.py            # metadata + HDR + time_base probing
```

## License

No license specified.
