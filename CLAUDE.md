# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PrismaSynth is a PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with TransNetV2 (GPU neural network), review/delete/split clips on a timeline, export as video, image sequence, or EDL. Single-track editor — no layers, no compositing.

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

**No tests or CI.** There is no test suite — verify changes by running the app.

## Architecture

Three-layer design: `src/ui/` (PySide6 widgets) → `src/core/` (business logic, threading) → `src/utils/` (ffprobe, paths).

**Entry point:** `src/main.py` → `MainWindow` in `src/ui/main_window.py` (the orchestrator that wires everything together). `main.py` adds `src/` to `sys.path`, sets dark Fusion theme, and installs an exception hook that writes to `src/crash.log`.

**Menu bar:** File, Edit, Timeline. Timeline menu contains Import, Export Video/Image/EDL, Detect Cuts, Play/Pause.

### Data Model

- `Clip` — references a source video by `source_id` + in/out frame numbers. `source_id=None` means it's a gap (empty space on timeline).
- `TimelineModel` (QObject) — ordered list of Clips/Gaps with selection tracking + undo/redo (50-level snapshot stack). Emits `clips_changed`, `selection_changed`, and `in_out_changed` signals. All position math uses integer frame numbers.
- `VideoSource` — immutable metadata (path, fps, dimensions, codec) for an imported video file.
- Project files (`.psynth`) are JSON containing sources + clips + playhead position + scroll offset + in/out points.

### Preview & Scrubbing (mpv/libmpv)

The preview widget embeds mpv for GPU-accelerated decode and display. The entire pipeline stays on GPU: NVDEC decode → GPU buffer → GPU display. No frames touch CPU RAM during scrubbing.

- `PreviewWidget` creates an mpv instance embedded in a QWidget via window handle (`wid`)
- Configured with `hwdec=auto`, `hr_seek=yes`, `keep_open=yes`, `ao=null`
- Timeline playhead changes → `mpv.command('seek', timestamp, 'absolute+exact')`
- Source switching: `mpv.loadfile(path)` when crossing clip boundaries (with `wait_for_property('seekable')`)
- Gaps: black overlay widget on top of mpv (no mpv state change — avoids stutter). Overlay hidden via `QTimer.singleShot(50ms)` after seek to prevent stale frame flash.
- mpv initialized in `showEvent()` after widget's winId() is valid
- **Zoom/Pan:** Scroll wheel zooms toward cursor (anchor-preserving math via `video-zoom` = `log2(r)`). Middle-mouse drags to pan. Pan clamped so video always covers the widget. Double-click resets to fit. Bottom-left `QComboBox` overlay shows presets (Fit, 50%-400%) and accepts arbitrary `N%` input. `_apply_zoom()` pushes `video-zoom`, `video-pan-x`, `video-pan-y` into mpv.
- **Seek throttle:** `seek_to_time()` enforces a minimum 33ms interval between seeks to prevent overwhelming mpv during rapid scrubbing.
- **Cached `_is_playing`:** A Python bool tracks playback state instead of querying `self._player.pause` (mpv C property). Updated by `play()`, `pause()`, `load_source()`. Avoids native property access on every mouse event.
- **`_ensure_video_visible()`** must be called before `loadfile` — `clear_frame()` sets `vid='no'` which disables the video track, and `wait_for_property('seekable')` will hang forever if the video track is disabled.

### Edit Modes & Keyboard Shortcuts

Two modes controlled by `EditMode` enum in `timeline_widget.py`:

- **Selection mode** (V key, default): Normal clip selection, marquee selection, playhead dragging.
- **Cut mode** (C key): Hover shows dashed cut-preview line, preview syncs to mouse position without moving playhead, click splits clip.
- **Quick-cut:** Right-click during playhead drag splits at playhead position (works in selection mode, no mode switch needed).

**WASD layout** (left hand in gaming position):

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| W | Delete (gap-leave) | E | Set In point |
| A | Select to gap | R | Set Out point |
| S | Split at playhead | X | Clear In/Out |
| D | Ripple delete | C | Cut mode |
| V | Selection mode | Space | Play/Pause |

### In/Out Render Points

- `TimelineModel._in_point` / `_out_point` (Optional[int] frame numbers) define export and scene detection range.
- `get_render_range()` returns `(start, end)` clamped to timeline bounds, filling missing ends with 0 / total-1.
- Visualized as cyan markers + semi-transparent dim overlay outside the range.
- Persisted in `.psynth` project files. Not included in undo/redo snapshots (tool setting, not timeline edit).

### Playback

Playback uses mpv natively (`player.pause = False`). A 60Hz QTimer (`_playback_timer`) syncs the timeline playhead to mpv's `time_pos`. At clip boundaries, contiguous same-source clips play through without interruption; only discontinuities trigger a seek. The `_playback_updating` flag prevents the playhead sync from triggering scrub logic.

**Gap playback:** Gaps are playable — preview shows black, playhead advances based on elapsed wall-clock time at the source FPS (no mpv involvement). State tracked via `_gap_start_time` / `_gap_start_frame`. When the gap ends, playback transitions to the next real clip by loading its source and calling `play()`. Check `_playback_timer.isActive()` (not just `is_playing`) to detect playback during gaps.

### Export Pipeline

**Architecture:** Parallel segment encoding via `ThreadPoolExecutor`, then concat with stream copy. Python is NOT in the data path for video export.

**Routing:** `_export_video()` → `_export_video_parallel()` for all codecs, with `_export_video_concat_legacy()` as fallback.

- **Segment coalescing:** `_build_source_groups()` groups clips by source and merges contiguous clips into single segments, reducing FFmpeg process count.
- **Parallel encoding:** `ThreadPoolExecutor` with as-soon-as-done scheduling. NVENC: 6 workers. CPU codecs: scales with `cpu_count // 4`, each with `-threads` for internal parallelism.
- **Single-segment bypass:** When coalescing produces 1 segment, encodes directly to output file (no temp dir, no concat).
- **HDR detection:** `probe_hdr()` in `ffprobe.py` checks `color_transfer` + `color_primaries`. SDR sources skip tonemap entirely.
- **SDR zero-copy (NVENC):** `-hwaccel_output_format cuda` + `scale_cuda=format=yuv420p` — frames never touch CPU.
- **HDR tonemap:** GPU path uses `tonemap_opencl=hable`, CPU fallback uses `zscale` chain. Output tagged with `-colorspace bt709 -color_trc bt709 -color_primaries bt709`.
- **Skip-same-res scale:** When output dimensions match source, the CPU `scale` filter is omitted entirely.
- **Frame accuracy:** `-frames:v N` (integer frame count) replaces time-based `-t` to eliminate float rounding. Pre-input `-ss` with `-accurate_seek` (default) for seek precision.
- **Codec presets:** H.264, H.265 (CPU + NVENC), ProRes 422 (`prores_aw` for profiles 0-3, `prores_ks` for 4444), FFV1 lossless (`-slices 4`). NVENC uses `-rc vbr -cq` (not `-crf`).
- **Exporter signals** (`progress`, `status`, `finished`, `error`) emitted from a `threading.Thread` — must use `Qt.ConnectionType.QueuedConnection` when connecting to UI slots.
- **Thread safety:** Process list guarded by `threading.Lock`, `failed` event for early abort on segment failure, `communicate()` for safe stderr handling.
- Image sequence export uses `_iter_frames_ffmpeg()` with per-source select filter + MJPEG pipe. Parallel frame writing via 4-worker ThreadPoolExecutor.
- Denoised export pipes decoded frames through FastDVDnet (5-frame sliding window) to FFmpeg encoder.

### EDL Export

CMX 3600 EDL for import into Premiere Pro, DaVinci Resolve, Avid. Options: include/exclude gaps, use in/out render range.

- **Time-based timecode conversion:** `frame/fps → actual time → NDF TC` using `round()`. This matches how Resolve derives timecodes from source PTS. Frame-counting (`frame // fps_int`) diverges at non-integer FPS (23.976, 29.97) by ~1 frame per 1000 frames.
- **Anchored durations:** SRC_OUT = SRC_IN_tc + duration (frame-counting from anchor). This preserves exact frame counts while SRC_IN uses time-based positioning.
- **Chained record timecodes:** REC_IN/OUT chain as a running TC total — no gaps between clips.
- Source path comments (`* SOURCE FILE:`) for NLE relinking.

### Thumbnail System

Viewport-prioritized thumbnail generation with playhead-distance sorting. 4 parallel ffmpeg `-ss` seeks (~38ms/frame).

- **Visible-only scope:** Only generates thumbnails for clips currently in the timeline viewport. No forward generation beyond visible area.
- **Playhead priority:** Within the visible set, frames closest to the playhead are generated first.
- **LQ proxy placeholders:** If a `.proxy` file exists (from scene detection), instantly upscales 48x27 frames to 192x108 as blurry placeholders while HQ thumbnails load.
- **4 parallel workers:** `ThreadPoolExecutor` with per-frame `-ss` seeks. Priority re-checked before each new submission (~100ms latency on viewport change).
- **Pause on scrub:** `_pause_event` halts generation during playhead drag. Resumes after 500ms idle (`_thumb_resume_timer`). Viewport reprioritize debounced to 300ms (`_viewport_timer`).
- **Memory-only cache:** `_mem_cache` dict of QImage at 192x108, keyed by `{source_id}_{frame_num}`. Regenerated fresh each session. `_ndarray_to_qimage` returns `.copy()` to prevent use-after-free across thread boundaries.
- **`communicate()`** for subprocess cleanup — prevents Windows handle leaks over hundreds of thumbnails.

### Proxy System (Scene Detection Only)

`ProxyFile` (48x27 mmap binary) is generated during scene detection for TransNetV2 input reuse. Stored at `%LOCALAPPDATA%/prismasynth/cache/proxies/`. Managed by `ProxyManager`. Also used as LQ thumbnail placeholders.

### Import & Scene Detection (separate steps)

1. **Import** (`ImportDialog` or drag-and-drop) — probes file via ffprobe, creates a single whole-file clip. No detection.
2. **Detect Cuts** (`DetectDialog`, Ctrl+D) — runs SceneDetector on a source, replaces its clips via `ripple_delete_by_source()`. Respects in/out render range when set.

Scene detection uses two decode paths: parallel 4-segment ffmpeg/NVDEC (~396 fps), single CPU fallback (~245 fps). TransNetV2 GPU inference with HSV frame differencing fallback. mpv and thumbnails are paused during detection to avoid GPU contention.

### Threading Model

| Thread | What | Key constraint |
|--------|------|---------------|
| Main (Qt) | All UI, signal slots, mpv commands | QPixmap only here |
| Thumbnail coordinator | 4 parallel ffmpeg `-ss` seeks per visible frame | Pauses during scrubbing, priority-sorted |
| Scene detection (QThread) | Parallel ffmpeg decode + TransNetV2 inference | Stores subprocess refs for cancellation |
| Export | Per-segment ffmpeg subprocesses (ThreadPoolExecutor) | Signals must use QueuedConnection to UI |

**Thread safety:** `threading.Lock` on VideoReader, FrameCache, and export process lists. Qt signals for cross-thread communication. ThumbnailCache emits `QImage` (thread-safe) not `QPixmap` (main-thread-only). Exporter uses `QueuedConnection` for all UI signal connections. Do NOT use `faulthandler.enable()` — it intercepts mpv's internal structured exceptions (`0xe24c4a02`) and kills the process.

### Gaps

W deletes a clip, leaving a gap (same-duration `Clip` with `source_id=None`). D ripple-deletes (collapses space). Adjacent gaps auto-merge. Gaps are selectable, visible as dark dashed rectangles. Skipped during export. Playable during playback (shows black, advances at source FPS). Deleting a gap (W or D) teleports the playhead to the gap's position.

### Timeline Widget

Custom-painted strip (not Qt model/view). Clip positions computed from cumulative frame counts via `_frame_to_pixel()` / `_pixel_to_frame()` to prevent truncation drift. Both account for `TIMELINE_H_PADDING`. Playhead clamped to content area. Track height is draggable (30-200px). Thumbnails scale to fill track height at 16:9 aspect with 6px color border. Marquee selection by dragging below the track (Ctrl to add). Timeline widget wrapped in a container with left/right margins.

**Paint order:** background → ruler → clips → in/out overlay → cut preview line → playhead → marquee rectangle.

## Key Conventions

- Keyboard shortcuts defined ONLY in `_setup_menus()` (not toolbar) to avoid Qt ambiguous shortcut conflicts.
- `TimelineModel.add_clips(assign_colors=False)` when loading from project file to preserve saved colors.
- `self._sources` dict is updated in-place (`clear()` + `update()`) never reassigned, because other objects hold references to it.
- Exporter stored as `self._exporter` on MainWindow to prevent garbage collection during background export.
- `_playback_updating` flag in MainWindow distinguishes playback-driven playhead updates from user scrubs (prevents playback from stopping itself).
- Never show downsampled proxy frames in the preview — user requires full-quality scrubbing via mpv at all times.
- On Windows, mmap'd proxy files cannot be overwritten — `ProxyFile.save_frames` catches `OSError` and skips gracefully.
- `TimelineModel.clear()` also clears the undo stack and in/out points (fresh start for new/load project).
- Mutation methods that modify `_clips` call `_push_undo()` after validation passes, before the actual mutation. Selection-only changes and in/out point changes are not undoable.
- `_on_new_project()` must stop the thumbnail cache before clearing readers/sources to avoid deadlocks.
- `load_source()` calls `_ensure_video_visible()` first — without this, `wait_for_property('seekable')` hangs if `vid='no'` was set by `clear_frame()`.
- Pause mpv and thumbnail cache before scene detection to prevent GPU contention crashes.
- All FFmpeg commands should include `-nostdin` (prevents stdin reads on Windows) and `-v error` (consistent error capture).
- Export uses `-frames:v N` (not `-t`) for frame-accurate output. Pre-input `-ss` for fast seeking.
- EDL timecodes use time-based conversion (`frame/fps`) with `round()`, not frame-counting (`frame // fps_int`). SRC_OUT is anchored to SRC_IN + duration for exact frame counts.
- Thumbnail subprocess must use `communicate()` (not `stdout.read()` + `wait()`) to prevent Windows handle leaks.
- `run.bat` sets `PYTHON_GIL=1` for Python 3.13+ free-threaded builds — required for correct threading behavior.
- `playback_engine.py` exists in `src/core/` but is unused — mpv handles playback natively. Retained as reference only.
- Autosave fires every 60 seconds when the project is dirty (has unsaved changes).
