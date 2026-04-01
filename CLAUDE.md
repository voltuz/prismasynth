# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PrismaSynth is a PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with TransNetV2 (GPU neural network), review/delete/split clips on a timeline, export as video or image sequence. Single-track editor — no layers, no compositing.

## Running

```bash
# Setup (first time)
python -m venv venv
venv\Scripts\pip install -r requirements.txt
# Install PyTorch with CUDA (for GPU scene detection)
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126

# Run
venv\Scripts\python src\main.py
# Or: run.bat
```

Requires FFmpeg + ffprobe on PATH. Python 3.11+.

## Architecture

Three-layer design: `src/ui/` (PySide6 widgets) → `src/core/` (business logic, threading) → `src/utils/` (ffprobe, paths).

**Entry point:** `src/main.py` → `MainWindow` in `src/ui/main_window.py` (the orchestrator that wires everything together).

### Data Model

- `Clip` — references a source video by `source_id` + in/out frame numbers. `source_id=None` means it's a gap (empty space on timeline).
- `TimelineModel` (QObject) — ordered list of Clips/Gaps with selection tracking. Emits `clips_changed` and `selection_changed` signals. All position math uses integer frame numbers.
- `VideoSource` — immutable metadata (path, fps, dimensions, codec) for an imported video file.
- Project files (`.psynth`) are JSON containing sources + clips + playhead position.

### Video Decode Stack

Three separate reader paths to avoid lock contention:

| Reader | Purpose | Obtained via |
|--------|---------|-------------|
| Scrub reader | Preview during scrubbing | `reader_pool.get_reader()` |
| Playback reader | Sequential playback | `reader_pool.get_playback_reader()` |
| Thumbnail readers | ffmpeg subprocess per frame | Not in reader pool |

All readers use PyAV with multi-threaded H.265 decode. Frame numbers use `Fraction` arithmetic (not float) for PTS conversion to avoid drift on long videos.

### Scrubbing Performance (the critical path)

Scrubbing uses a cascading fallback for instant response:

1. **FrameCache** (LRU, 500 frames in RAM) — exact hit = 0ms
2. **ProxyFile** (48x27 mmap binary, all frames) — 0.003ms per frame
3. **Nearest cached frame** (tolerance ±60) — approximate, instant
4. **Background full-res decode** (PyAV seek) — 150-200ms for 4K HEVC

The proxy file is generated automatically during scene detection (TransNetV2 already decodes every frame at 48x27). Stored at `%LOCALAPPDATA%/prismasynth/cache/proxies/`.

`ScrubDecoder` debounces requests (8ms), skips stale decodes, and pre-decodes ±15 frames around the playhead after each seek.

### Threading Model

| Thread | What | Key constraint |
|--------|------|---------------|
| Main (Qt) | All UI, signal slots | QPixmap only here |
| Playback prefetch | Sequential decode into ring buffer (60 frames) | Separate reader instance |
| Scrub decode | Single background decode + pre-decode ring | Checks `_pending_frame` to abort stale work |
| Thumbnail coordinator | Dispatches parallel ffmpeg `-ss` seeks (4 workers) | Pauses during scrubbing, yields between emissions |
| Scene detection (QThread) | TransNetV2 inference + ffmpeg frame extraction | Stores subprocess ref for cancellation |
| Export | FFmpeg pipe encoding | Uses playback reader (not scrub reader) |

**Thread safety:** `threading.Lock` on VideoReader and FrameCache. Qt signals for cross-thread communication (auto-queued). ThumbnailCache emits `QImage` (thread-safe) not `QPixmap` (main-thread-only).

### Scene Detection

Primary: TransNetV2 (`transnetv2-pytorch`) — dilated 3D CNN, F1=96.2%, runs on CUDA. Frames extracted via ffmpeg subprocess at 48x27 with `-hwaccel cuda`. Inference in sliding 100-frame windows (step 50).

Fallback: HSV frame differencing via PyAV + OpenCV (no GPU required).

### Gaps

Deleting a clip leaves a gap (same-duration `Clip` with `source_id=None`). Adjacent gaps auto-merge. Deleting a gap collapses the space. Gaps are selectable, visible as dark dashed rectangles. Skipped during export and playback.

### Timeline Widget Pixel Math

Clip positions computed from cumulative frame counts: `start_px = int(cumulative_frames * pixels_per_frame)`. This prevents the per-clip `int()` truncation drift that would desync the playhead from clip positions over hundreds of clips. The `_frame_to_pixel()` method is the single source of truth.

## Key Conventions

- Keyboard shortcuts defined ONLY in `_setup_menus()` (not toolbar) to avoid Qt ambiguous shortcut conflicts.
- `TimelineModel.add_clips(assign_colors=False)` when loading from project file to preserve saved colors.
- `self._sources` dict is updated in-place (`clear()` + `update()`) never reassigned, because PlaybackEngine and ScrubDecoder hold references to it.
- Exporter stored as `self._exporter` on MainWindow to prevent garbage collection during background export.
