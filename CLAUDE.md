# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PrismaSynth is a PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with TransNetV2 (GPU neural network), review/delete/split clips on a timeline, export as video, image sequence, FCPXML, or OpenTimelineIO. Single-track editor — no layers, no compositing.

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

## Development tooling (`scripts/`)

No unit-test framework, but three dev scripts exercise real code paths for correctness and perf regression tracking. All take `--video PATH` (repeatable) and support `--skip SECTION`.

- **`scripts/system_check.py`** — correctness harness. Runs the real `Exporter` class via a `QCoreApplication` event loop, checks frame-accuracy against source, FCPXML frame-exact fractions, MOV timebase integrity, NVENC `scale_cuda` output dims, data-model fuzz invariants. Hard-fails on regression. ~70s end-to-end with two sample videos. **Run before committing export-pipeline changes.**
- **`scripts/perf_benchmark.py`** — perf timing harness. Exercises decode / thumbnail / scene-detection / export speed via hand-rolled FFmpeg commands (NOT through `Exporter`, so doesn't validate production correctness — use `system_check.py` for that). ~110s.
- **`scripts/cut_inspect.py`** — mpv-embedded Qt GUI. Load an exported video, visually mark cut boundaries with the scrubber, extract ±2 frames around each cut for frame-by-frame inspection. Useful for diagnosing visual anomalies at clip transitions.
- **`scripts/export_diag.py`** — one-shot CLI that runs a single segment through multiple ffmpeg config variants (two-stage seek, single seek, CPU decode, NVENC, etc.) and reports first-frame hashes. Useful for isolating ffmpeg-config-specific export issues.

## Architecture

Three-layer design: `src/ui/` (PySide6 widgets) → `src/core/` (business logic, threading) → `src/utils/` (ffprobe, paths).

**Entry point:** `src/main.py` → `MainWindow` in `src/ui/main_window.py` (the orchestrator that wires everything together). `main.py` adds `src/` to `sys.path`, sets dark Fusion theme, and installs an exception hook that writes to `src/crash.log`.

**Menu bar:** File, Edit, Timeline. Timeline menu contains Import, Export Video/Image/XML, Detect Cuts, Play/Pause.

### Data Model

- `Clip` — references a source video by `source_id` + in/out frame numbers. `source_id=None` means it's a gap (empty space on timeline).
- `TimelineModel` (QObject) — ordered list of Clips/Gaps with selection tracking + undo/redo (50-level snapshot stack). Emits `clips_changed`, `selection_changed`, and `in_out_changed` signals. All position math uses integer frame numbers.
- `VideoSource` — immutable metadata (path, fps, dimensions, codec) for an imported video file.
- Project files (`.psynth`) are JSON containing sources + clips + playhead position + scroll offset + in/out points.

### Preview & Scrubbing (mpv/libmpv)

The preview widget embeds mpv for GPU-accelerated decode and display. The entire pipeline stays on GPU: NVDEC decode → GPU buffer → GPU display. No frames touch CPU RAM during scrubbing.

- `PreviewWidget` creates an mpv instance embedded in a QWidget via window handle (`wid`)
- Configured with `hwdec=auto`, `hr_seek=yes`, `hr_seek_framedrop=yes`, `keep_open=yes`, `ao=null`, 150MB/75MB demuxer forward/backward cache
- Timeline playhead changes → `mpv.command_async('seek', timestamp, 'absolute+exact')` — non-blocking so the Qt main thread never stalls during 4K H.265 seeks. mpv abandons stale seeks when a new one arrives.
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
- **HDR detection:** `probe_hdr()` in `ffprobe.py` checks `color_transfer` + `color_primaries`. SDR sources skip tonemap entirely. Cached on the group dict (`group["is_hdr"]`) so it runs once per source, not per segment.
- **SDR zero-copy (NVENC):** `-hwaccel_output_format cuda` + `scale_cuda={w}:{h}:format=yuv420p` — the entire pipeline (NVDEC → scale_cuda → NVENC) stays on GPU, including downscales like 4K→1080p. `_build_vf` returns `None` for SDR+NVENC so the GPU filter chain in `_build_segment_cmd` takes over.
- **HDR tonemap:** GPU path uses `tonemap_opencl=hable:desat=0:peak=1000` (peak in cd/m²), CPU fallback uses `zscale` chain with `tonemap=hable:desat=0:peak=10` (peak relative to npl=100). Output tagged with `-colorspace bt709 -color_trc bt709 -color_primaries bt709`. Explicit `peak` avoids per-segment auto-detection "ramp-up" — curve is movie-wide, derived from the stream's declared color space.
- **HDR metadata safety:** Filter chain prepends `setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc` so `tonemap_opencl` has consistent signalling even when the seek lands on a mid-GOP P/B frame whose per-frame metadata is missing — otherwise the first frame of that clip can bypass the tonemap.
- **Skip-same-res scale:** When output dimensions match source, the CPU `scale` filter is omitted entirely.
- **Frame accuracy:** `-frames:v N` (integer frame count) replaces time-based `-t` to eliminate float rounding. Seek timestamps go through `Exporter._frame_to_seek_ts(frame, fps)` which returns `(frame - 0.5) / fps` — the half-frame margin defeats IEEE 754 off-by-one when `src_in / fps` rounds microscopically above the target frame's true PTS, causing ffmpeg's accurate-seek to drop the target frame and emit `target+1`.
- **Two-stage seek:** pre-input `-ss` lands ~1s before target (keyframe-accurate, cheap via NVDEC); post-input `-ss 1.0` does the final second accurately. The post-input stage is load-bearing — it pushes ~24 frames through the filter graph before the first output frame, warming up the OpenCL tonemap context. Without it, the first frame of some clips bypasses tonemap entirely.
- **`-fps_mode passthrough` is required** — without it, a filter-graph init race causes some clips' first output frame to skip the tonemap filter.
- **`-video_track_timescale 24000000`** on both per-segment encode AND the concat command — gives clean integer frame durations for all broadcast/cinema rates (23.976: 1001000 ticks, 29.97: 800800, 24/30/60: also integer). The default 1/16000 timebase stores jittering 672/656 durations that NLEs like Resolve interpret strictly and display as duplicate frames. Temp-file extension matches the output extension so the muxer option actually lands (MKV ignores it).
- **`setpts=PTS-STARTPTS`** at the top of CPU filter chains — rebases each segment's first frame to PTS=0 so the stream-copy concat appends cleanly. Not used on the zero-copy GPU path (scale_cuda chain).
- **Codec presets:** H.264, H.265 (CPU + NVENC), ProRes 422 (`prores_aw` for profiles 0-3, `prores_ks` for 4444), FFV1 lossless (`-slices 4`). NVENC uses `-rc vbr -cq` (not `-crf`).
- **Exporter signals** (`progress`, `status`, `finished`, `cancelled`, `error`) emitted from a `threading.Thread` — must use `Qt.ConnectionType.QueuedConnection` when connecting to UI slots. `cancelled` distinguishes user-abort from exception so the dialog shows "Export cancelled" instead of "Error".
- **Subprocess tracking:** All `Popen` instances go through `Exporter._register_proc()` / `_unregister_proc()` under `self._procs_lock`. `cancel()` reads the list under lock and kills every live process — safe across parallel, single-segment, legacy-concat, and image-sequence paths. The image-sequence decoder wraps its loop in `try/finally` to kill+wait the subprocess even if the consumer raises mid-iteration.
- **Concat drain threads:** legacy path spawns a thread per segment that calls `proc.communicate()` — drains stderr concurrently to prevent the classic `wait() + stderr.read()` deadlock when ffmpeg fills a stderr pipe buffer.
- Image sequence export uses `_iter_frames_ffmpeg()` with per-source select filter + rawvideo pipe. Parallel frame writing via 4-worker ThreadPoolExecutor.

### XML Export (FCPXML 1.9)

Only supported timeline-interchange format. FCPXML encodes positions as rational fractions (`frames * 1001/24000s` for NTSC 23.976), so there is no timecode-to-frame conversion on the importer side. Resolve seeks the source file by time, and an accurate time maps 1:1 to the correct source frame. EDL was removed — its CMX 3600 timecode strings forced Resolve through a lossy NDF interpretation that drifted ±1 frame with no reliable client-side fix.

- **Use the TRUE frame duration per rate:** `_rate_to_frame_duration(fps)` returns `(1001, 24000)` for 23.976, `(1001, 30000)` for 29.97, `(1, 25)` for 25, etc. We briefly tried emitting `(1, 24)` for 23.976 to "match" Resolve's internal 23.976/24 naming conflation — this broke the frame mapping: `start="2769/24s"` = 115.375s, but true frame 2769 is at 115.490s, so Resolve's time-based seek loaded file frame 2766 (−3 drift). The correct fraction is `2769*1001/24000s` which Resolve time-seeks to exactly file frame 2769. Confirmed empirically — see `scripts/system_check.py::section_export_xml`.
- **Asset-clip `start` nudge (`_src_seek_str`):** every asset-clip `start` gets a `+frame_num/2` added to its numerator — for NTSC 23.976 that's `(N*1001 + 500)/24000s` instead of `(N*1001)/24000s`. *Why:* Resolve rewrites imported NTSC assets as `frameDuration="1/24s"` internally and rounds our `start` to the nearest `1/24` tick. For `N mod 1000 < 500`, that tick falls just below frame N's PTS, and Resolve's time-based seek loads frame N−1 ("first frame from previous clip" drift, observed on ~half the clips of test_08.psynth). The +500 numerator lands `start` mid-way through frame N's PTS range — well inside the range for either rounding direction. Duration and offset do NOT get nudged (they're timeline positions/lengths, not source seeks). Verified drift-free on all 16 clips of test_08.psynth.
- **Frame-exact fractions:** `_time_str(frames, num, den)` emits `(frames*num)/den s` with a common denominator so every time in the document is an integer multiple of the frame duration. Asset-clip `start` uses `_src_seek_str` which adds the NTSC-rounding nudge above.
- **`<media-rep>` child:** source URI goes in a `<media-rep kind="original-media" src="..."/>` child element on the asset, not an `src` attribute — matches Resolve's own export convention.
- **File URIs:** `pathlib.Path.as_uri()` produces percent-encoded `file:///C:/...` URLs — already safe for direct XML embedding (no `&`, `<`, `>`, or `"` after encoding).
- **Spine layout:** one `<asset-clip>` per real clip. `offset` = cumulative timeline position (packed if `include_gaps=False`, absolute-in-render-range if `True`). `start` = source in-point as a fraction. Gaps (when included) emit `<gap>` elements.
- **Asset reuse:** one `<asset>` per *source* file (not per clip), IDs assigned in discovery order as `r2`, `r3`, etc. (`r1` is reserved for the `<format>`).
- **Timeline name from filename:** `title=` parameter defaults to None; when None, derives `os.path.splitext(os.path.basename(output_path))[0]` so Resolve's "Load XML" dialog defaults the Timeline name and Import-timeline dropdown to the file's name instead of a generic literal. The derived value also feeds `<event name="…">` (the bin folder Resolve creates for imported assets). Pass explicit `title=` to override.
- **Remaining ±1 drift trace:** source-file container timebase, not this exporter. MOVs exported with pre-v0.2.0 PrismaSynth use `time_base=1/16000`, which can't represent 23.976 exactly (667 vs 668 tick frames averaging). Re-export the source with current PrismaSynth (`-video_track_timescale 24000000`) for clean frame math end-to-end.

### OTIO Export (OpenTimelineIO native JSON)

`core.otio_exporter.export_otio` writes an `.otio` document matching the public OTIO schema (Timeline.1 / Stack.1 / Track.1 / Clip.2 / Gap.1 / ExternalReference.1 / TimeRange.1 / RationalTime.1). Resolve Studio imports it natively; Premiere goes through an external OTIO adapter.

- **No `opentimelineio` Python dep:** we emit the JSON directly (same pattern as `xml_exporter.py`). OTIO 0.18.1 has a packaging bug on Python 3.14 — wheel tagged `cp314` but ships `cp313` `.pyd` files — and we only ever wrote files, so the library was a nicety, not a requirement.
- **`source_range.start_time` nudge (`_SRC_SEEK_NUDGE = 0.25`):** every clip's `source_range.start_time.value` is written as `frame + 0.25`. *Why:* at rates like 24000/1001, the round-trip `time = value / our_rate; frame = floor(time * reader_rate)` drifts by ±1 ULP and can land `N.0` at `N-1.9999999...`. Verified empirically against Resolve (test_09.otio, clip 5 source_start 3543 → 3542 drift with no nudge). Sweeping frames 0-10000 with Resolve's rate (`23.976023976023979` vs our `23.976023976023978`) drifts 692 frames (~7%). A `+0.25` nudge eliminates all drift under `floor()`, `trunc()`, and `round()` reader modes. FCPXML uses `+0.5` (mid-1/24-tick for its different rounding mechanism); OTIO needs `+0.25` because `+0.5` breaks readers using `round()` — residues just over 0.5 snap to N+1. Only `source_range.start_time` is nudged: duration, `available_range`, and `Gap.source_range` stay integer.
- **Clip.2 schema:** `media_references` dict + `active_media_reference_key="DEFAULT_MEDIA"`. Clip.1 (`media_reference` singular) is also legal but older; Clip.2 is shipped since OTIO 0.15 (2021) and supported by every current adapter.
- **`available_range`** on each `ExternalReference` = `(start=0, duration=source.total_frames)` at sequence rate. Asserts source length so OTIO-aware tools can display source handles without re-probing the file.
- **Asset reuse:** one `ExternalReference` *dict* per source file shared by dict identity across clips. After JSON round-trip those become equal-but-distinct dicts — still semantically correct, just no longer pointer-shared.
- **Metadata round-trip:** `metadata["prismasynth"]` stores `source_id` / `clip_id` / `color_index` / `label` so a future OTIO importer could reconstruct a .psynth project with no loss. `metadata["Resolve_OTIO"]` (effects, locks, timeline TC offset) is never written by us — Resolve fills it on its own round-trip.
- **Timeline name from filename:** `title=` defaults to None; when None, derives from the output filename's basename (extension stripped). Lands on `Timeline.name` in the OTIO doc. Pass explicit `title=` to override. Symmetric with FCPXML's behaviour — don't diverge the two paths.
- **Smoke test:** `scripts/otio_smoke.py` builds a minimal in-memory timeline and asserts schema shape, nudge presence, integer counts for duration/available_range/gaps, metadata round-trip, packed-layout behaviour, render-range clipping, and the empty-timeline guard.

### Thumbnail System

Persistent, long-lived thumbnail generator with sweep-based sequential decode. Created once per session, survives timeline edits. CPU-only decode (no NVDEC) to avoid GPU contention with mpv scrubbing.

- **Visible-only scope:** Only generates thumbnails for clips currently in the timeline viewport. No forward generation beyond visible area.
- **Playhead sweep:** From the playhead, sweeps right (forward) then left (backward) through sorted frame targets. Within each sweep, frames within `_SEQUENTIAL_THRESHOLD` (400 frames) are decoded sequentially via PyAV (`_grab_frames_sweep`) — avoids redundant re-seeks from the same keyframe. Distant frames fall back to independent ffmpeg seeks (`_grab_frame_single`).
- **LQ proxy placeholders:** If a `.proxy` file exists (from scene detection), instantly upscales 48x27 frames to 192x108 as blurry placeholders while HQ thumbnails load.
- **6 parallel workers:** Persistent `ThreadPoolExecutor`. Sweep runs and single-frame seeks are both submitted to the pool. Each sweep opens one PyAV container, seeks once, and decodes forward through all targets.
- **Persistent coordinator thread:** Runs for the session. `notify_clips_changed()` rebuilds the clip lookup (which frames to generate) without destroying the cache or thread pool — already-cached HQ thumbnails for unchanged frames reuse instantly. `reprioritize()` updates viewport priority on scroll.
- **Pause on scrub:** `_pause_event` halts generation during playhead drag. Resumes after 500ms idle (`_thumb_resume_timer`). `_wake_event` signals the coordinator when there's new work. Viewport reprioritize debounced to 300ms (`_viewport_timer`).
- **Memory-only cache:** `_mem_cache` dict of QImage at 192x108, keyed by `{source_id}_{frame_num}`. Persists across clip changes (splits, deletes). `.copy()` on QImage to prevent use-after-free across thread boundaries.
- **GPU isolation:** Thumbnails use CPU-only ffmpeg (no `-hwaccel cuda`) to leave NVDEC exclusively for mpv scrubbing. This prevents GPU contention that would stall the preview during scrubbing.
- **Cache Thumbnails dialog:** `ui.cache_thumbnails_dialog.CacheThumbnailsDialog` is a user-triggered modal that bakes the on-disk thumbnail cache for the first AND last frame of every non-gap clip on the timeline (or only clips inside the in/out range when set). Backed by `thumbnail_cache.BulkCacheJob`, which runs the same 6-worker `ThreadPoolExecutor` as the live cache but on every targeted clip boundary instead of just the visible viewport. Reusable across runs in one session: after Done / Cancelled / Already-Cached the dialog returns to idle with Start re-enabled so the user can re-run with the Overwrite checkbox toggled. Use this before exporting a project with many clips to pre-warm the disk cache.

### Proxy System (Scene Detection Only)

`ProxyFile` (48x27 mmap binary) is generated during scene detection for TransNetV2 input reuse. Stored at `%LOCALAPPDATA%/prismasynth/cache/proxies/`. Managed by `ProxyManager`. Also used as LQ thumbnail placeholders. Supports `frame_offset` (stored in `.offset` sidecar file) for proxies that don't start at source frame 0 — enables partial-range detection. `get_frame(source_frame)` subtracts the offset internally.

### Import & Scene Detection (separate steps)

1. **Import** (`ImportDialog` or drag-and-drop) — multi-file select, probes all files, creates one whole-file clip per source appended sequentially. All files in a batch must share the same resolution and FPS. If the timeline already has sources, new imports must match.
2. **Detect Cuts** (`DetectDialog`, Ctrl+D) — runs SceneDetector on **all non-gap clips** on the timeline. Two detector backends are exposed via a dropdown:
   - **TransNetV2** (default) — fast, hard-cut neural network. Per-segment loop. HSV differencing as automatic fallback if TransNetV2 fails to load.
   - **OmniShotCut** — slower transformer alternative. Despite OmniShotCut's full output covering dissolves/fades/wipes/pushes/slides/zooms/doorways/sudden-jumps, **PrismaSynth filters its results to hard cuts only** (inter_label == 1). Soft transitions are deliberately skipped — a 1-second dissolve between two shots produces NO cut, leaving one merged clip on the timeline. Runs in a sidecar subprocess (see below). Set `hard_cuts_only: false` in the segment header if you ever want the full transition set re-enabled.
   Each clip's source segment is processed independently; cuts are forced at source boundaries. Replaces analyzed clips via `TimelineModel.replace_detected()`. Partially-clamped clips (by in/out range) get prefix/suffix clips preserved. Re-detects everything in the analysis range.

Scene detection decode fallback chain (shared by both detectors via `core.ffmpeg_decode.decode_to_array`): (1) parallel 4-segment ffmpeg with `scale_cuda` (NVDEC + GPU resize), (2) parallel 4-segment ffmpeg with CPU scale (NVDEC + CPU resize, ~396 fps), (3) single ffmpeg CPU fallback (~245 fps). Each path is probed before use. mpv and thumbnails are paused during detection to avoid GPU contention.

### OmniShotCut Sidecar Architecture

OmniShotCut requires `torch==2.5.1` / Python 3.10 / CUDA 12.4 — incompatible with PrismaSynth's Python 3.13+ free-threaded + `torch cu126`. To avoid the version conflict, OmniShotCut runs in a separate venv (`venv-omnishotcut/` at repo root, gitignored) as a sidecar subprocess.

- **Setup** — `scripts/setup_omnishotcut.py` is a one-shot installer that uses `uv` (downloaded as a static binary to `.uv/uv.exe`) to install Python 3.10, create the sidecar venv, install torch + OmniShotCut requirements, clone OmniShotCut to `third_party/OmniShotCut/`, download the model checkpoint to `%LOCALAPPDATA%/prismasynth/models/OmniShotCut_ckpt.pth`, run a `--selftest`, and write a `venv-omnishotcut/.prismasynth_ready` sentinel file. Idempotent. Re-runnable with `--repair`. Triggered from the Detect Cuts dialog: when OmniShotCut is selected and the sentinel is missing, the "Detect" button is replaced with "Set up OmniShotCut" → opens `OmnishotcutSetupDialog` which tails the script's stdout in a `QPlainTextEdit`.
- **Sidecar** — `scripts/omnishotcut_sidecar.py` is the in-venv script. Loads the model ONCE per Detect Cuts run (5-15s) and processes all segments sequentially, amortising the load cost. Communicates with the main process over JSON-line stdio + raw frame bytes. Defines `_single_array_inference()` — a fork of OmniShotCut's `single_video_inference` that takes a pre-decoded numpy array instead of a video file path, so frames decoded by the parent's NVDEC pipeline at the model's required resolution can be piped in directly. Per-window progress callback emitted during inference.
- **Runner** — `core.omnishotcut_runner.OmnishotcutRunner` is the main-process wrapper. Spawns the sidecar with `CREATE_NEW_PROCESS_GROUP` (Windows) so cancellation can send `CTRL_BREAK_EVENT` for graceful shutdown before `proc.kill()`. Decodes each segment in-process via `decode_to_array`, sends a JSON segment header + raw bytes to the sidecar, drains the sidecar's stdout in a background thread (analyzing-progress callbacks dispatched live, results enqueued for the main thread). Closes stdin to signal end of work. SceneDetector's `cancel()` propagates to the runner.
- **Cut mapping** — OmniShotCut returns shot ranges `[[start, end_exclusive], ...]` per segment plus per-shot intra/inter labels. The sidecar's `_ranges_to_cuts(pred_ranges, inter_labels, total, hard_cuts_only=True)` emits a cut at the end of shot `i` only when `inter_labels[i+1] == 1` (hard_cut). Transitions (label 3), transition_source (2), sudden_jump (4), and new_start chunk markers (0) are all skipped — soft transitions produce no cut, leaving the surrounding shots as one merged clip. The resulting cut list goes through the same `_cuts_to_clips()` `+1` shift as TransNetV2.
- **Setup detection** — `is_setup_complete()` checks all three of: sentinel file, sidecar python.exe, and `third_party/OmniShotCut/` exist. The sentinel is written as the LAST step of setup so a partially-installed venv (e.g. setup killed mid-install) is correctly reported as not-ready.

### Project Portability & Relinking

`.psynth` files store both an absolute `file_path` and a `relative_path` (relative to the project file's directory) per source. On load, if the absolute path is missing, `core.project.load_project()` silently falls back to the relative path resolved against the project file's directory — sources that travel together with the project (e.g. zipped to a USB stick) reopen with no UI prompt.

For sources that still don't resolve, `_load_from()` shows `ui.relink_dialog.RelinkDialog` BEFORE clearing the previously-open project's state — cancelling the dialog leaves the previous project intact (no destructive load). The dialog uses folder pickers per row: the user picks the folder containing the missing source, the dialog finds the basename inside it (case-insensitive on Windows). Picking one row's folder runs `_folder_rebase()` across the other still-missing rows to auto-link any matching basename. Auto-found rows pass STRICT silent validation (exact width/height, fps within 0.02, frame count within ±1%); explicit Browse uses LENIENT validation with a warn-and-override dialog on mismatch.

On accept, `_load_from()` updates each relinked `VideoSource`'s metadata from the dialog's probe cache (file_path + width/height/fps/total_frames + codec + audio fields), clamps any clip whose `source_out` exceeds the new (smaller) total_frames with a single batch warning, then clears state and proceeds with the existing load flow. Sources whose path still doesn't exist (user clicked Skip) bypass `_reader_pool.register_source()` and `_proxy_manager.load_or_open()` — clips referencing them render black with no thumbnails, matching existing missing-file behaviour. Relinks set `self._dirty = True` so the new paths persist on next autosave.

`save_project()` wraps `os.path.relpath` in `try/except ValueError` to handle Windows cross-drive paths (stored as `relative_path: null`). `load_project()` uses `sd.get("relative_path")` so legacy `.psynth` files without the field load unchanged.

### Threading Model

| Thread | What | Key constraint |
|--------|------|---------------|
| Main (Qt) | All UI, signal slots, mpv commands | QPixmap only here |
| Thumbnail coordinator | Persistent thread + 6-worker pool, sweep decode via PyAV + ffmpeg | Pauses during scrubbing, playhead-distance sweep |
| Scene detection (QThread) | Parallel ffmpeg decode + TransNetV2 inference (OR drives OmniShotCut sidecar) | Stores subprocess refs for cancellation |
| OmniShotCut sidecar | Separate Python 3.10 process; loads model once, infers per segment | Communicates over JSON-line stdio + raw bytes; cancel via CTRL_BREAK_EVENT then kill |
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
- Export uses `-frames:v N` (not `-t`) for frame-accurate output. Pre-input `-ss` timestamps go through `Exporter._frame_to_seek_ts()` (half-frame margin) — never use `src_in / fps` directly.
- Export temp-file extension matches the output extension (`.mov` for ProRes, `.mp4` for H.264/5, `.mkv` for FFV1). Hardcoding `.mkv` would silently drop muxer options like `-video_track_timescale` on non-Matroska outputs.
- Export subprocesses MUST be registered via `Exporter._register_proc()`. Never write to `self._active_procs` directly.
- `ExportDialog` cancel button is state-driven: "Cancel" (idle, closes dialog) → "Cancel Export" (running, emits `cancel_requested` signal) → "Close" (done/finished, closes). `Exporter.cancelled` signal transitions the dialog to the "done" state with "Export cancelled." status.
- Timeline interchange: FCPXML (`core.xml_exporter.export_fcpxml`) and OTIO (`core.otio_exporter.export_otio`). FCPXML asset-clip `start` uses `_src_seek_str` with a `+frame_num/2` numerator nudge (NTSC `1/24` tick rounding). OTIO `source_range.start_time.value` uses `+_SRC_SEEK_NUDGE (0.25)` (float ULP round-trip drift). Different root causes, different nudge values — don't cross-port. Don't revert either to unnudged seek positions.
- `TimelineModel.set_in_point()` / `set_out_point()` reject invalid inputs (would make In >= Out) instead of silently wiping the opposite marker — in/out changes are not undoable, so a silent wipe destroys data.
- `TimelineModel.add_clips([])` and `replace_detected({})` early-return without pushing undo (would clobber the redo stack on empty ops like a failed import).
- `save_project()` writes to `filepath + ".tmp"` then `os.replace()` — atomic, so the 60s autosave can't destroy the project mid-write.
- `PreviewWidget.load_source()` passes `timeout=5.0` to `wait_for_property('seekable')` — without a timeout, a corrupt or unreachable file would hang the UI thread.
- Thumbnail subprocess must use `communicate()` (not `stdout.read()` + `wait()`) to prevent Windows handle leaks.
- `run.bat` sets `PYTHON_GIL=1` for Python 3.13+ free-threaded builds — required for correct threading behavior.
- `playback_engine.py` exists in `src/core/` but is unused — mpv handles playback natively. Retained as reference only.
- Autosave fires every 60 seconds when the project is dirty (has unsaved changes).
- `TimelineStrip` emits `scrub_started` / `scrub_ended` signals on playhead drag. Connected to `PreviewWidget.scrub_start()` / `scrub_end()` and used to pause/resume thumbnails.
- `TimelineModel.replace_detected(replacements)` swaps clips by ID with detected sub-clip lists — the core integration point after scene detection. Preserves gaps and non-matched clips.
- `ThumbnailCache` is created once and lives for the session. `_start_thumbnail_cache()` creates it on first call, subsequent calls use `notify_clips_changed()` + `reprioritize()`. Only `_on_new_project()` destroys it.
- `ProxyManager.load_or_open(source, force_reopen=True)` is used after detection to pick up newly saved proxy files. Without `force_reopen`, cached proxies from import time would be returned.
- `_load_from()` runs the relink dialog BEFORE clearing state. Do not reorder — cancelling the dialog must leave the previous project intact. Skipped sources (user-checked "Skip") must bypass `_reader_pool.register_source()` and `_proxy_manager.load_or_open()`.
- `save_project()` writes `relative_path` per source (relative to the `.psynth` directory) alongside the absolute path. Cross-drive paths on Windows (`os.path.relpath` raises `ValueError`) get stored as `relative_path: null`.
- `RelinkDialog` per-row Browse uses `QFileDialog.getExistingDirectory`, not `getOpenFileName` — the user picks a folder and the dialog finds the basename inside. Don't switch back to file pickers; the folder UX is what users wanted.
- FCPXML / OTIO `export_*(... title=None ...)` derive the title from `output_path`. Don't restore the old `title="PrismaSynth"` literal default — it overrides what users see in Resolve.
- OmniShotCut runs in **hard-cuts-only mode** by default: `_ranges_to_cuts()` filters by `inter_labels[i+1] == 1 (hard_cut)`. Soft transitions (dissolve/fade/wipe/push/slide/zoom/doorway) and sudden_jump are deliberately skipped per user preference. To re-enable the full transition set, send `hard_cuts_only: false` in the per-segment JSON header — there is currently no UI toggle for this.
- OmniShotCut sidecar is launched ONCE per Detect Cuts run, not per segment — model load is 5-15s and must be amortised. `OmnishotcutRunner` interleaves decode+send+wait sequentially per segment; do not refactor to per-segment subprocess spawning.
- `core.omnishotcut_runner.is_setup_complete()` checks the sentinel file (`venv-omnishotcut/.prismasynth_ready`) AND the venv python AND `third_party/OmniShotCut/`. The sentinel is written as the LAST step of `scripts/setup_omnishotcut.py` — checking just python.exe would race with mid-setup.
- The decode helpers in `core.ffmpeg_decode` (`decode_to_array` and friends) are SHARED by both detector backends. Don't fork them per detector — width/height parameters already make them detector-agnostic.
- OmniShotCut has no LICENSE file in its upstream repo — vendored snapshot at `third_party/OmniShotCut/` is gitignored. Personal-use only until upstream clarifies licensing; do not redistribute PrismaSynth bundles that include a checkpoint or vendored copy.
- `decode_to_array()` returns `(padded_array, n_frames, method, elapsed_secs)`. The padded array's leading and trailing slack is left UNINITIALIZED — caller must zero or pad-replicate it if the detector requires deterministic edges (e.g. TransNetV2 replicates first/last frame).
