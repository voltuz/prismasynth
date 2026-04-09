# PrismaSynth

A PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with a GPU neural network, review/trim/split clips on a timeline, and export as video, image sequence, or EDL.

Single-track editor — no layers, no compositing. Built for speed: the entire preview pipeline stays on GPU (NVDEC decode → GPU buffer → mpv display), and export runs parallel FFmpeg segments with zero-copy NVENC for SDR sources.

## Features

- **GPU scrubbing** — libmpv-backed preview with `hwdec=auto` and exact seeks. No frames touch CPU RAM while scrubbing.
- **Scene detection** — TransNetV2 GPU inference with 4-segment parallel NVDEC decode (~396 fps). HSV frame differencing fallback when CUDA is unavailable.
- **Parallel export** — per-segment FFmpeg workers via ThreadPoolExecutor, then concat with stream copy. NVENC SDR path is zero-copy.
  - H.264 / H.265 (CPU and NVENC)
  - ProRes 422 (profiles 0-3) and 4444
  - FFV1 lossless
  - Image sequence (PNG/JPEG)
  - Denoised export through FastDVDnet (5-frame sliding window)
- **HDR handling** — `probe_hdr()` detects HDR sources and routes through GPU `tonemap_opencl=hable` or a CPU `zscale` fallback. SDR sources skip tonemap entirely.
- **Frame-accurate** — export uses `-frames:v N` (integer frame count) to eliminate float rounding from time-based durations.
- **CMX 3600 EDL export** — time-based timecode conversion matching DaVinci Resolve's PTS derivation. Compatible with Premiere Pro, Resolve, and Avid.
- **Viewport-prioritized thumbnails** — only visible clips get HQ thumbnails, sorted by playhead distance. 48x27 proxy placeholders upscale instantly while HQ loads.
- **In/Out render range** — cyan markers define the export and scene-detection window.
- **Undo/redo** — 50-level snapshot stack on the timeline model.
- **Project files** — `.psynth` JSON containing sources, clips, playhead, scroll, and in/out points.

## Keyboard Shortcuts

WASD layout (left hand in gaming position):

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| W | Delete (leave gap) | E | Set In point |
| A | Select to gap | R | Set Out point |
| S | Split at playhead | X | Clear In/Out |
| D | Ripple delete | C | Cut mode |
| V | Selection mode | Space | Play/Pause |

Right-click while dragging the playhead performs a quick-cut at the playhead position without switching modes.

## Requirements

- Python 3.11+
- Windows (tested on Windows 11)
- FFmpeg and ffprobe on `PATH`
- `libmpv-2.dll` in `src/` — grab it from an [mpv.io](https://mpv.io/installation/) dev build
- NVIDIA GPU with CUDA 12.6+ for scene detection and NVENC export (optional — CPU fallback paths exist)

## Setup

```bash
git clone https://github.com/voltuz/prismasynth.git
cd prismasynth

python -m venv venv
venv\Scripts\pip install -r requirements.txt

# PyTorch with CUDA (for GPU scene detection)
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Place `libmpv-2.dll` in `src/`.

## Running

```bash
venv\Scripts\python src\main.py
```

Or double-click `run.bat`.

## Workflow

1. **Import** a video (File → Import, drag-and-drop, or Ctrl+I) — creates one whole-file clip.
2. **Detect Cuts** (Ctrl+D) — replaces the clip with one clip per shot. Respects the in/out render range when set.
3. **Review** — scrub, delete bad shots with `W`, ripple-delete with `D`, split with `S`, set cut mode with `C`.
4. **Export** — Timeline menu → Export Video / Image Sequence / EDL.

## Architecture

Three-layer design:

```
src/ui/     PySide6 widgets (timeline, preview, dialogs)
src/core/   Business logic, threading, export pipeline, scene detection
src/utils/  ffprobe wrappers, path helpers
```

Entry point is `src/main.py` → `MainWindow` in `src/ui/main_window.py`, which wires everything together.

### Data Model

- `VideoSource` — immutable metadata (path, fps, dimensions, codec) for an imported file.
- `Clip` — references a source by `source_id` + in/out frame numbers. `source_id=None` means it's a gap.
- `TimelineModel` — ordered list of clips/gaps with selection, undo/redo, and in/out render points. Emits Qt signals on mutation.

All position math uses integer frame numbers. Cumulative frame counts are converted to pixels via `_frame_to_pixel()` / `_pixel_to_frame()` to prevent truncation drift.

### Threading

| Thread | Work | Constraint |
|--------|------|-----------|
| Main (Qt) | UI, signal slots, mpv commands | QPixmap main-thread only |
| Thumbnail coordinator | 4 parallel `ffmpeg -ss` workers | Pauses during scrubbing |
| Scene detection (QThread) | Parallel FFmpeg decode + TransNetV2 | Stores subprocess refs for cancellation |
| Export (ThreadPoolExecutor) | Per-segment FFmpeg subprocesses | Signals via `QueuedConnection` |

mpv and thumbnail generation are paused during scene detection to avoid GPU contention.

## Project Layout

```
src/
├── main.py                  # entry point
├── ui/
│   ├── main_window.py       # orchestrator
│   ├── timeline_widget.py   # custom-painted timeline strip
│   ├── preview_widget.py    # embedded mpv
│   └── ...
├── core/
│   ├── timeline_model.py    # clips, selection, undo/redo
│   ├── exporter.py          # parallel FFmpeg pipeline
│   ├── scene_detector.py    # TransNetV2 + NVDEC
│   ├── thumbnail_cache.py   # viewport-prioritized generation
│   ├── fastdvdnet/          # denoiser
│   └── project.py           # .psynth save/load
└── utils/
    ├── ffprobe.py           # HDR detection, metadata probing
    └── paths.py
```

## License

No license specified.
