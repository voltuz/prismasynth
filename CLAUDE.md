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

Requires FFmpeg + ffprobe on PATH, `libmpv-2.dll` in `src/` (from mpv.io builds). Python 3.11+.

## Architecture

Three-layer design: `src/ui/` (PySide6 widgets) → `src/core/` (business logic, threading) → `src/utils/` (ffprobe, paths).

**Entry point:** `src/main.py` → `MainWindow` in `src/ui/main_window.py` (the orchestrator that wires everything together).

### Data Model

- `Clip` — references a source video by `source_id` + in/out frame numbers. `source_id=None` means it's a gap (empty space on timeline).
- `TimelineModel` (QObject) — ordered list of Clips/Gaps with selection tracking + undo/redo (50-level snapshot stack). Emits `clips_changed` and `selection_changed` signals. All position math uses integer frame numbers.
- `VideoSource` — immutable metadata (path, fps, dimensions, codec) for an imported video file.
- Project files (`.psynth`) are JSON containing sources + clips + playhead position.

### Preview & Scrubbing (mpv/libmpv)

The preview widget embeds mpv for GPU-accelerated decode and display. The entire pipeline stays on GPU: NVDEC decode → GPU buffer → GPU display. No frames touch CPU RAM during scrubbing.

- `PreviewWidget` creates an mpv instance embedded in a QWidget via window handle (`wid`)
- Configured with `hwdec=auto`, `hr_seek=yes`, `keep_open=yes`, `ao=null`
- Timeline playhead changes → `mpv.command('seek', timestamp, 'absolute+exact')`
- Source switching: `mpv.loadfile(path)` when crossing clip boundaries (with `wait_for_property('seekable')`)
- Gaps: black overlay widget on top of mpv (no mpv state change — avoids stutter). Overlay hidden via `QTimer.singleShot(50ms)` after seek to prevent stale frame flash.
- mpv initialized in `showEvent()` after widget's winId() is valid

### Playback

Playback uses mpv natively (`player.pause = False`). A 60Hz QTimer (`_playback_timer`) syncs the timeline playhead to mpv's `time_pos`. At clip boundaries, contiguous same-source clips play through without interruption; only discontinuities trigger a seek. The `_playback_updating` flag prevents the playhead sync from triggering scrub logic.

### Export (FFmpeg with HDR→SDR tone mapping)

Export decodes each clip segment via ffmpeg subprocess (not PyAV) with a `zscale→tonemap=hable→zscale` filter chain for proper HDR/Dolby Vision → SDR conversion. Both video export and image sequence export use this path.

### Proxy System

Two proxy tiers exist for thumbnail extraction:

- **ProxyFile** (48x27 mmap binary) — generated during scene detection, stored at `%LOCALAPPDATA%/prismasynth/cache/proxies/`
- **JpegProxyFile** (960x540 JPEG-indexed) — generated in background by `HQProxyGenerator`, stored next to video file as `.jproxy`. Format: fixed header + (N+1) uint64 offset table + concatenated JPEG blobs.
- `ProxyManager` prefers `.jproxy` over `.proxy` when both exist

Thumbnails (`ThumbnailCache`) extract from the JPEG proxy when available (sub-millisecond), falling back to ffmpeg `-ss` seeks. 96x54 JPEGs cached to disk. `_ndarray_to_qimage` returns `.copy()` to prevent use-after-free across thread boundaries.

### Import & Scene Detection (separate steps)

1. **Import** (`ImportDialog` or drag-and-drop) — probes file via ffprobe, creates a single whole-file clip. No detection.
2. **Detect Cuts** (`DetectDialog`, Ctrl+D) — runs SceneDetector on a source, replaces its clips via `ripple_delete_by_source()`.

Scene detection uses three decode paths in priority order: JPEG proxy reuse (~628 fps), parallel 4-segment ffmpeg/NVDEC (~396 fps), single CPU fallback (~245 fps). TransNetV2 GPU inference with HSV frame differencing fallback. Progress bar resets between decode and inference stages.

### Threading Model

| Thread | What | Key constraint |
|--------|------|---------------|
| Main (Qt) | All UI, signal slots, mpv commands | QPixmap only here |
| Thumbnail coordinator | Dispatches parallel ffmpeg `-ss` seeks (4 workers) | Pauses during scrubbing, yields between emissions |
| Scene detection (QThread) | Parallel ffmpeg decode + TransNetV2 inference | Stores subprocess refs for cancellation |
| HQ proxy generator | Background ffmpeg decode + JPEG encode | Emits `finished` signal when done |
| Export | Per-segment ffmpeg subprocess pipe | Uses ffmpeg directly (not PyAV) |

**Thread safety:** `threading.Lock` on VideoReader and FrameCache. Qt signals for cross-thread communication (auto-queued). ThumbnailCache emits `QImage` (thread-safe) not `QPixmap` (main-thread-only).

### Gaps

Backspace deletes a clip, leaving a gap (same-duration `Clip` with `source_id=None`). Delete key ripple-deletes (collapses space). Adjacent gaps auto-merge. Gaps are selectable, visible as dark dashed rectangles. Skipped during export and playback. Preview shows black overlay on gaps.

### Timeline Widget

Custom-painted strip (not Qt model/view). Clip positions computed from cumulative frame counts via `_frame_to_pixel()` to prevent truncation drift. Track height is draggable (30-200px). Thumbnails scale to fill track height at 16:9 aspect with 6px color border. Marquee selection by dragging below the track (Ctrl to add).

## Key Conventions

- Keyboard shortcuts defined ONLY in `_setup_menus()` (not toolbar) to avoid Qt ambiguous shortcut conflicts.
- `TimelineModel.add_clips(assign_colors=False)` when loading from project file to preserve saved colors.
- `self._sources` dict is updated in-place (`clear()` + `update()`) never reassigned, because other objects hold references to it.
- Exporter stored as `self._exporter` on MainWindow to prevent garbage collection during background export.
- `_playback_updating` flag in MainWindow distinguishes playback-driven playhead updates from user scrubs (prevents playback from stopping itself).
- Never show downsampled proxy frames in the preview — user requires full-quality scrubbing via mpv at all times.
- On Windows, mmap'd proxy files cannot be overwritten — `ProxyFile.save_frames` catches `OSError` and skips gracefully.
- `TimelineModel.clear()` also clears the undo stack (fresh start for new/load project).
- Mutation methods that modify `_clips` call `_push_undo()` first. Selection-only changes are not undoable.
