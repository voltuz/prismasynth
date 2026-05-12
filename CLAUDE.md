# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PrismaSynth is a PySide6 video editing tool for curating deepfake training datasets. Import movies, auto-detect shot boundaries with TransNetV2 (GPU neural network), review/delete/split clips on a timeline, tag clips with People (groups), and export selected groups as video, audio, image sequence, FCPXML, or OpenTimelineIO. Single-track editor — no layers, no compositing.

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
- **`scripts/cut_inspect.py`** — mpv-embedded Qt GUI launchable from Tools → Cut Inspect (or standalone). Load an exported video, visually mark cut boundaries with the scrubber, extract ±2 frames around each cut for frame-by-frame inspection. Useful for diagnosing visual anomalies at clip transitions. The Tools-menu launcher spawns it in a separate Python process so its own mpv instance can't conflict with the preview's.
- **`scripts/export_diag.py`** — one-shot CLI that runs a single segment through multiple ffmpeg config variants (two-stage seek, single seek, CPU decode, NVENC, etc.) and reports first-frame hashes. Useful for isolating ffmpeg-config-specific export issues.
- **`scripts/audio_drift_check.py`** — drives a real ProRes-MOV embedded export of a `.psynth`, ffprobes the result, and compares the total audio sample count against the rational expected count. Reports drift in samples and milliseconds. Pre-fix baseline on the BillyCrystal project showed ~11 ms / segment audio shortfall; post-fix it reads 0.000 ms exactly. Run after any change to the per-segment audio chain in `core/exporter.py`.

## Architecture

Three-layer design: `src/ui/` (PySide6 widgets) → `src/core/` (business logic, threading) → `src/utils/` (ffprobe, paths).

**Entry point:** `src/main.py` → `MainWindow` in `src/ui/main_window.py` (the orchestrator that wires everything together). `main.py` adds `src/` to `sys.path`, sets dark Fusion theme, and installs an exception hook that writes to `src/crash.log`.

**Menu bar:** File, Edit, Timeline, Tools.
- File: New / Open / Save / Save As / Recent Projects / Keyboard Shortcuts.
- Edit: Split / Delete / Ripple Delete / Undo / Redo / Select All / Select to Gap / Set In/Out / Selection Mode / Cut Mode / Scrub Follow.
- Timeline: Import, Export Video / Image Sequence / Audio Only / XML / OTIO, Detect Cuts, Play/Pause.
- Tools: Relink…, Cache Manager…, Cut Inspect….

### Data Model

- `Clip` — references a source video by `source_id` + in/out frame numbers. `source_id=None` means it's a gap (empty space on timeline). Carries `group_ids: list[str]` (People memberships).
- `Group` (in `core/group.py`) — a People tag: `name`, `color` (hex), optional `digit` (0-9 bound to a keyboard shortcut), `id` (UUID). Project-scoped; persists to `.psynth`.
- `TimelineModel` (QObject) — ordered list of Clips/Gaps with selection tracking + undo/redo (50-level snapshot stack). Owns the groups dict (`_groups: Dict[str, Group]`). Snapshot deep-copies clips so the new mutable `group_ids` list survives undo/redo. Emits `clips_changed`, `selection_changed`, `in_out_changed`, `groups_changed`. All position math uses integer frame numbers.
- `VideoSource` — immutable metadata (path, fps, dimensions, codec, audio_codec, audio_sample_rate, audio_channels) for an imported video file.
- Project files (`.psynth`) are JSON containing sources + clips (with group_ids) + groups + playhead position + scroll offset + `pixels_per_frame` (zoom) + in/out points + orphan paths.

### Preview & Scrubbing (mpv/libmpv)

The preview widget embeds mpv for GPU-accelerated decode and display. The entire pipeline stays on GPU: NVDEC decode → GPU buffer → GPU display. No frames touch CPU RAM during scrubbing.

- `PreviewWidget` creates an mpv instance embedded in a QWidget via window handle (`wid`)
- Configured with `hwdec=auto`, `hr_seek=yes`, `hr_seek_framedrop=yes`, `keep_open=yes`, 150MB/75MB demuxer forward/backward cache
- **Audio output is enabled** — mpv's default `ao=auto` plays system audio during real-time playback. Bottom-left preview overlay has a mute QToolButton + volume QSlider (session-only state, not persisted).
- Timeline playhead changes → `mpv.command_async('seek', timestamp, 'absolute+exact')` — non-blocking so the Qt main thread never stalls during 4K H.265 seeks. mpv abandons stale seeks when a new one arrives.
- Source switching: `mpv.loadfile(path)` when crossing clip boundaries (with `wait_for_property('seekable')`)
- Gaps: black overlay widget on top of mpv (no mpv state change — avoids stutter). Overlay hidden via `QTimer.singleShot(50ms)` after seek to prevent stale frame flash.
- mpv initialized in `showEvent()` after widget's winId() is valid
- **Zoom/Pan:** Scroll wheel zooms toward cursor (anchor-preserving math via `video-zoom` = `log2(r)`). Middle-mouse drags to pan. Pan clamped so video always covers the widget. Double-click resets to fit. Bottom-left `QComboBox` overlay shows presets (Fit, 50%-400%) and accepts arbitrary `N%` input. `_apply_zoom()` pushes `video-zoom`, `video-pan-x`, `video-pan-y` into mpv.
- **Seek throttle:** `seek_to_time()` enforces a minimum 33ms interval between seeks to prevent overwhelming mpv during rapid scrubbing.
- **Cached `_is_playing`:** A Python bool tracks playback state instead of querying `self._player.pause` (mpv C property). Avoids native property access on every mouse event.
- **`_ensure_video_visible()`** must be called before `loadfile` — `clear_frame()` sets `vid='no'` which disables the video track, and `wait_for_property('seekable')` will hang forever if the video track is disabled.

### Customizable Keyboard Shortcuts

`core/shortcuts.py::ShortcutManager` is the single source of truth for every user-rebindable shortcut. The `SHORTCUTS` registry has ~40 entries grouped by category (File / Edit / Timeline / People). Each entry has a stable `id`, display `name`, and `default` key sequence.

- MainWindow's `_setup_menus` calls `self._shortcut_mgr.attach_action(sid, qaction)` instead of `setShortcut(...)` so the user's chosen key (loaded from QSettings on startup) is applied at QAction creation time.
- Timeline-widget keys (arrow stepping, Home/End, Up/Down clip nav, digit 0-9 group toggle) are wired as `QShortcut` objects with `WidgetWithChildrenShortcut` context on the timeline widget — so digit keys don't fire when the user is typing in the People panel name field.
- `File → Keyboard Shortcuts…` opens `KeyboardShortcutsDialog` (in `ui/shortcuts_dialog.py`): a categorised tree, double-click a row → modal that captures the next non-modifier key. Conflicts (key already bound elsewhere) are rejected with `QMessageBox.warning("'X' is already used by 'Y'.")` — uniqueness is enforced.
- ShortcutManager.set_key returns `None` on success or the conflicting action's display name. Reset Row / Reset All / Clear semantics live in the dialog.
- Persistence: `QSettings` keys under `shortcuts/<id>`. On startup, ShortcutManager detects collisions (e.g. a new release adds a default that clashes with a user override) and clears one side so the unique-key invariant holds.

### Edit Modes

Two modes controlled by `EditMode` enum in `timeline_widget.py`:

- **Selection mode** (V key by default): Normal clip selection, marquee selection, playhead dragging.
- **Cut mode** (C key by default): Hover shows dashed cut-preview line, preview syncs to mouse position without moving playhead, click splits clip.
- **Quick-cut:** Right-click during playhead drag splits at playhead position (works in selection mode, no mode switch needed).

Default WASD layout (left hand, gaming position) for the most-used edits — every key is rebindable via the Keyboard Shortcuts dialog.

### People (Groups)

Project-scoped tag system for clips, used to curate "person A vs person B vs untagged" subsets for deepfake training.

- A `Group` has a name, color, optional digit (0-9). Clips can belong to multiple groups simultaneously (`Clip.group_ids: list[str]`).
- `core.shortcuts.SHORTCUTS` registers digit shortcuts `group_digit_1` … `group_digit_9` then `group_digit_0` (keyboard number-row order). Pressing the digit toggles the digit's group on every selected clip; if no group holds that digit, an inline `QInputDialog.getText` prompt creates one and immediately tags the selection.
- Toggle semantics on multi-select: if every targeted clip already has the group, removes from all; otherwise adds to those that don't (so all become members). Implemented in `TimelineModel.toggle_clip_group`.
- The right column hosts a `QTabWidget` with two tabs (vertical text tabs, `TabPosition.West`): **Clip Info** (existing per-clip metadata panel) and **People** (the group registry).
- `ui/people_panel.py::PeoplePanel` shows one row per group: color swatch (clickable for `QColorDialog`), inline-editable name, digit combo (`— / 1 / 2 / … / 9 / 0`, conflict-rejected), live clip count, trash-icon delete (with confirmation that includes the count of clips that'll lose the tag).
- The timeline strip reserves 42px below each clip body (`GROUP_LABEL_HEIGHT * GROUP_LABEL_MAX_VISIBLE = 14 * 3`) for up to three group chips per clip — name + group color + W3C-luminance-based readable text. `+N` overlay on the last visible chip when a clip belongs to more groups than fit.
- Persistence: `core.project.save_project` writes `groups` array; load filters stale `group_ids` so a clip referencing a deleted group doesn't crash.
- Snapshot/restore covers groups (deep-copied), so add/remove/rename/recolour/digit-bind are all undoable.

### Project Versions

Durable cross-session rollback layer that sits *above* the in-memory undo stack. Every `.psynth` file gets a sibling `<project>.psynth.versions/` directory containing timestamped copies plus an `index.json` manifest. Lives in `core/project_versions.py`; UI is `ui/versions_dialog.py`; wired into MainWindow via `_ensure_version_store`, `_take_pre_op_snapshot`, and the autosave timer.

- `ProjectVersionStore(project_path)` is the single API surface — `create(trigger, label=None)`, `list_versions()`, `restore(filename)`, `delete(filename)`, `prune()`. Bound to a single project path; MainWindow stores the current instance in `self._version_store` and rebinds on save / load via `_ensure_version_store()`.
- **Seven valid triggers** baked into the filename (sans `autosave` for readability) and the manifest: `autosave`, `manual`, `pre_detect_cuts`, `pre_multi_delete`, `pre_group_delete`, `pre_source_removal`, `pre_restore`. Anything else logs a warning and returns None.
- **Pre-op snapshot points** in `main_window.py`: `_do_remove_source` (source removal), `_on_delete`/`_on_ripple_delete` when >5 clips are selected, post-detect just before `replace_detected` swaps in detected clip lists, `PeoplePanel.group_delete_confirmed` signal (emitted by PeoplePanel after the user confirms the delete prompt but *before* the timeline mutation — connected in MainWindow.__init__), and inside `_on_restore_version` itself.
- **Flush-then-snapshot semantics:** `_take_pre_op_snapshot` first calls `_save_to(self._project_path)` if the project is dirty, then `store.create(trigger=...)`. So the snapshot reflects the working state in front of the user, not whatever was last autosaved. Same dance happens inside `_on_manual_project_snapshot` and `_on_restore_version`.
- **Filename shape:** `YYYY-MM-DD_HH-MM-SS[_<trigger>][_<label-slug>].psynth`. Same-second collisions get `_2`, `_3`, … suffix appended. Label is `re`-slugified with non-alphanumerics → `-` and 40-char cap.
- **Retention (`prune()`):** keep newest `_KEEP_NEWEST` (= 50) unconditionally; for older entries bucket by age — last 1 day = one-per-hour buckets, last 1 week = one-per-day buckets, older = one-per-ISO-week buckets — and keep the newest in each bucket. Runs after every `create()`.
- **Self-healing list (`list_versions`):** reconciles `index.json` against `*.psynth` on disk every call. Manifest entries for deleted files are dropped; orphan files (user copied a snapshot in by hand) are re-derived via `_derive_entry()` parsing the filename, then the manifest is rewritten. **`create()` deliberately does NOT use `list_versions()` to read the manifest** — it calls `_read_manifest()` directly, because `list_versions()` would re-derive the freshly-copied file as an orphan and then `manifest.insert(0, entry)` below would dupe it.
- **Restore flow:** `VersionsDialog.restore_requested(filename)` → MainWindow `_on_restore_version` → flush dirty + take `pre_restore` snapshot → `_load_from(version_path, project_path_override=original_path)` → mark dirty. The new `project_path_override` param keeps the live `_project_path` pointing at the user's real `.psynth` while loading the version contents in-memory. Marked dirty so the next autosave persists the restored state through.
- **All ops are best-effort:** every filesystem operation in `project_versions.py` catches and logs; the user's primary `.psynth` save path is never blocked or compromised when the versions dir is read-only / out of space / permissioned away. Failures inside `_take_pre_op_snapshot` log via `logger.exception` and never propagate.
- **Manifest format:** `{"versions": [{filename, timestamp (ISO-8601, second precision), trigger, label, clip_count, source_count, size_bytes}, …]}`. Written atomically via `tmp + os.replace`.
- **Action enable/disable:** `_refresh_version_actions()` greys out `File → Project Versions…` and `File → Snapshot Project Now…` until the project has a path on disk (i.e. has been saved once). `_refresh_snapshot_action()` does the same for `Tools → Snapshot Frame` and additionally requires the playhead to be over a non-gap clip.

### Frame Snapshot

User-triggered PNG capture of the current preview frame. `Tools → Snapshot Frame` (F12) — bound via `core.shortcuts.SHORTCUTS` id `snapshot_frame`.

- `core.frame_snapshot.snapshot_frame_to_png(source_path, frame_num, fps, out_path)` runs a one-shot `ffmpeg -ss <ts> -i <src> -frames:v 1 -update 1 <out>` with `subprocess.CREATE_NO_WINDOW` on Windows and a 30s timeout.
- `_seek_ts()` reuses the exporter's `(frame_num - 0.5) / fps` half-frame margin — same IEEE-754 rationale as `Exporter._frame_to_seek_ts`. The requested `frame_num` is the first frame ffmpeg keeps after the accurate seek, so the PNG matches the preview pixel-for-pixel.
- **Output dir:** `<project_stem>_snapshots/` next to the `.psynth`. Created on first use via `os.makedirs(..., exist_ok=True)`.
- **Filename:** `{source_stem}_f{source_frame_num}_{HHMMSS}.png`. The source frame number is computed via `TimelineModel.timeline_frame_to_source_frame(playhead)` so the same frame from different clips of the same source produces the same filename modulo the timestamp.
- **Disabled state:** action greys out when the project has no path yet (output dir would be ambiguous) or the playhead is over a gap / past the timeline (no source frame to capture). `_refresh_snapshot_action()` recomputes on every `_on_playhead_frame` and on the project lifecycle hooks (`_on_clear_project`, `_on_save_project`, `_on_load_project`).
- **Best-effort:** ffmpeg failures get logged and the status bar shows "Snapshot failed — see logs"; nothing is raised.

### Tools Menu

- **Relink…** — opens `RelinkDialog` with **all** imported sources (not just missing). Per-row Browse repoints any source. The right-click-on-Media-Pool relink is the per-source variant; this is the global view. After OK, `_apply_relink_results` updates VideoSource metadata, clamps clip out-points if the new source is shorter, re-registers reader+proxy, and re-extracts the source thumbnail.
- **Cache Manager…** — `ui/cache_manager_dialog.py`. Modal dialog showing per-category disk usage: Source Thumbnails (`cache/source_thumbs/`), Proxy Files (`cache/proxies/`), Disk Thumbnails (`cache/thumbs/`), OmniShotCut Model (`models/OmniShotCut_ckpt.pth`). Each row has a Clear button (with confirmation that shows the size that'll be freed). Includes a Refresh button and an Open Folder button (`os.startfile(get_cache_dir().parent)` opens `%LOCALAPPDATA%/prismasynth/`). Releases proxy mmap locks via `proxy_manager.close_all()` before clearing the proxies category — Windows otherwise refuses to delete in-use mmap'd files.
- **Cut Inspect…** — spawns `scripts/cut_inspect.py` as a detached subprocess (`subprocess.DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows). Isolated from the main app's mpv instance to prevent conflicts.

### In/Out Render Points

- `TimelineModel._in_point` / `_out_point` (Optional[int] frame numbers) define the render range.
- `get_render_range()` returns `(start, end)` clamped to timeline bounds, filling missing ends with 0 / total-1.
- Visualized as cyan markers + semi-transparent dim overlay outside the range. The In marker line renders at `_frame_to_pixel(in_pt)` (left edge of the In frame); the Out marker line renders at `_frame_to_pixel(out_pt + 1)` (right edge of the Out frame). When only Out is set, the In marker is drawn implicitly at frame 0 so the highlighted region is bookended on both sides.
- Persisted in `.psynth` project files. Not included in undo/redo snapshots (tool setting, not timeline edit).
- **Collision-nudge on E / R:** pressing E (Set In) at the Out frame (or at `out_pt + 1` — the visible Out marker line, which is one pixel past Out) shoves Out forward by 1 so the gesture sets In here without losing Out. Mirrored: pressing R at the In frame shoves In back by 1. Boundary rejected silently when the nudge would push past the timeline tail.

### Playback

Playback uses mpv natively (`player.pause = False`). A 60Hz QTimer (`_playback_timer`) syncs the timeline playhead to mpv's `time_pos`. At clip boundaries, contiguous same-source clips play through without interruption; only discontinuities trigger a seek. The `_playback_updating` flag prevents the playhead sync from triggering scrub logic.

**Gap playback:** Gaps are playable — preview shows black, playhead advances based on elapsed wall-clock time at the source FPS (no mpv involvement). State tracked via `_gap_start_time` / `_gap_start_frame`. When the gap ends, playback transitions to the next real clip by loading its source and calling `play()`. `_toggle_play` checks both `_preview.is_playing` AND `_playback_timer.isActive()` — Space pauses correctly during gap playback (mpv stays paused but the wall-clock timer drives the playhead).

### Export Pipeline

**Architecture:** Parallel segment encoding via `ThreadPoolExecutor`, then concat with stream copy. Python is NOT in the data path for video export.

**Unified dialog:** `ui/export_dialog.py::ExportDialog` is a single modal hosting **five tabs**: Video / Image Sequence / Audio only / XML / OTIO. Two shared checkboxes above the tabs apply to whichever export the user runs:

- **Include gaps between clips** — when checked, gap clips render as black video frames + silent audio (Video / Image Seq / Audio Only) or `<gap>` markers (XML / OTIO). Default unchecked.
- **Use in/out render range** — disabled when no in/out is set; otherwise enabled and checked by default. Unchecking exports the full timeline ignoring the in/out range.

The standalone XmlDialog and OtioDialog were removed in v0.13.0; their menu entries (`Export XML…`, `Export OTIO…`) route into the unified dialog at `tab=3` / `tab=4`.

**Audio modes** (Video tab dropdown): `embedded` (default — embedded in container), `none` (`-an`), `standalone` (audio-only output, video controls disabled), `both` (video with embedded audio + sidecar audio file). Audio formats for standalone: WAV (PCM), FLAC, MP3 (192k), M4A (AAC 320k). Sidecar location: "Next to video file" (auto-derived basename) or "Custom path".

**Group filter** (above the tabs, alongside the gap/range checkboxes): `ui/group_filter_widget.py::GroupFilterWidget` shows checkboxes for every project group plus an `(Untagged)` row. No box ticked = no filter (export everything, default). Any tick = filter active; clip exports iff `core.group.clip_matches_filter(clip, filter)` returns True. Gaps are governed by Include-gaps alone — the group filter applies only to non-gap clips. The shared widget is reused by all tabs so the filter is set once per dialog session.

**Routing:**
- `_export_video()` → `_export_video_parallel()` for all codecs, with `_export_video_concat_legacy()` as fallback.
- `audio_mode == "standalone"` → `_export_audio_only()`. `"both"` → video pipeline + `_extract_audio_sidecar()` post-step.
- `mode == "xml"` / `"otio"` (handled in MainWindow's `_run_export`) → `core.xml_exporter.export_fcpxml` / `core.otio_exporter.export_otio`.

**Segment building:** `_build_segments` walks `timeline.clips` once; honours `self._use_render_range` and `self._include_gaps` (set per-export from the settings dict). Gap clips emit a synthetic segment `(None, 0, count, fps, None)` when `include_gaps` is True; downstream cmd-builders render them via `lavfi color=c=black:s=WxH:r=FPS` + `anullsrc`. The People-group filter applies inside this loop too. **Segments are passed to `_export_video_parallel` in TIMELINE order** so concat output is correctly ordered (this also fixes a latent ordering quirk for non-monotonic timelines).

**Encoder details (largely unchanged):**

- **Segment coalescing:** `_build_source_groups()` previously coalesced contiguous same-source clips. v0.13.0 dropped coalescing in the parallel path in favour of timeline-order iteration; the perf hit is small and timeline-order correctness is worth it.
- **Parallel encoding:** `ThreadPoolExecutor` with as-soon-as-done scheduling. NVENC: 6 workers. CPU codecs: scales with `cpu_count // 4`, each with `-threads` for internal parallelism.
- **HDR detection:** `probe_hdr()` in `ffprobe.py` checks `color_transfer` + `color_primaries`. SDR sources skip tonemap entirely. Cached on a per-source params dict so it runs once per source, not per segment.
- **SDR zero-copy (NVENC):** `-hwaccel_output_format cuda` + `scale_cuda={w}:{h}:format=yuv420p` — the entire pipeline (NVDEC → scale_cuda → NVENC) stays on GPU.
- **HDR tonemap:** GPU path uses `tonemap_opencl=hable:desat=0:peak=1000`, CPU fallback uses `zscale` chain with `tonemap=hable:desat=0:peak=10`. Output tagged with `-colorspace bt709 -color_trc bt709 -color_primaries bt709`.
- **HDR metadata safety:** Filter chain prepends `setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc` so `tonemap_opencl` has consistent signalling even when the seek lands on a mid-GOP P/B frame.
- **Skip-same-res scale:** When output dimensions match source, the CPU `scale` filter is omitted entirely.
- **Frame accuracy:** `-frames:v N` (integer frame count) replaces time-based `-t`. Seek timestamps go through `Exporter._frame_to_seek_ts(frame, fps) = (frame - 0.5) / fps` — half-frame margin defeats IEEE 754 off-by-one.
- **Two-stage seek:** pre-input `-ss` lands ~1s before target (keyframe-accurate, cheap via NVDEC); post-input `-ss 1.0` does the final second accurately. The post-input stage is load-bearing — it warms up the OpenCL tonemap context.
- **`-fps_mode passthrough` is required** — without it, a filter-graph init race causes some clips' first output frame to skip the tonemap filter.
- **`-video_track_timescale 24000000`** on both per-segment encode AND the concat command — gives clean integer frame durations for all broadcast/cinema rates. Temp-file extension matches the output extension.
- **`setpts=PTS-STARTPTS`** at the top of CPU filter chains — rebases each segment's first frame to PTS=0 so the stream-copy concat appends cleanly.
- **Audio is encoded in a SEPARATE per-segment ffmpeg pass** (since v0.14.x). The video pass strips audio (`-an`); a parallel audio-only pass uses single-stage input `-ss` + `aresample=48000` + `aformat=...stereo` + `atrim=end_sample={Exporter._exact_audio_samples(count, fps)}` to produce a sample-precise WAV; a third stream-copy mux pass joins them. Why not one pass? The two-stage seek's output `-ss post_ss` interacts with `-frames:v` and chops the audio decoder's flush partway, producing ~half-a-frame audio shortage per segment that compounds to seconds across 100+ clips. The fix is verified by `scripts/audio_drift_check.py` (drift goes from ~11 ms/segment to 0). DO NOT collapse the audio back into the video command — the two-stage seek for HDR OpenCL warmup and the audio sample-count enforcement need separate seek strategies. `Exporter._exact_audio_samples(count, fps, sample_rate=48000)` uses rational integer math (NTSC `count * 1001 * 48000 // 24000`) so cumulative sample count is exactly correct over arbitrary segment counts.
- **Codec presets:** H.264, H.265 (CPU + NVENC), ProRes 422 (`prores_aw` for profiles 0-3, `prores_ks` for 4444), FFV1 lossless (`-slices 4`). NVENC uses `-rc vbr -cq` (not `-crf`).
- **Exporter signals** (`progress`, `status`, `finished`, `cancelled`, `error`) emitted from a `threading.Thread` — must use `Qt.ConnectionType.QueuedConnection` when connecting to UI slots.
- **Subprocess tracking:** All `Popen` instances go through `Exporter._register_proc()` / `_unregister_proc()`. `cancel()` reads the list under lock and kills every live process.
- **Gap rendering:** `_build_segment_cmd` (parallel + legacy) and `_export_audio_only` each have a gap branch using `lavfi color` (video) / `anullsrc` (audio). `_iter_frames_ffmpeg` (image sequence) yields `np.zeros((height, width, 3), dtype=np.uint8)` for each gap frame at the end of the source-group iteration.
- **Audio-only path:** `_export_audio_only` builds per-segment WAV temp files (PCM, easy to concat losslessly), then a single concat-and-encode pass to the user's chosen format.
- **Sidecar audio (audio_mode='both'):** `_extract_audio_sidecar` runs `ffmpeg -i {video_output} -vn -c:a {fmt}` after the video pipeline succeeds.

### XML Export (FCPXML 1.9)

- **Use the TRUE frame duration per rate:** `_rate_to_frame_duration(fps)` returns `(1001, 24000)` for 23.976, `(1001, 30000)` for 29.97, `(1, 25)` for 25, etc.
- **Asset-clip `start` nudge (`_src_seek_str`):** every asset-clip `start` gets a `+frame_num/2` added to its numerator. Resolve rewrites NTSC asset frame durations as `1/24s` internally and rounds our `start` to the nearest tick; the +500 numerator nudge lands `start` mid-way through the target frame's PTS range and avoids the "first frame from previous clip" drift.
- **Frame-exact fractions:** `_time_str(frames, num, den)` emits `(frames*num)/den s` with a common denominator so every time in the document is an integer multiple of the frame duration.
- **`<media-rep>` child:** source URI goes in a `<media-rep kind="original-media" src="..."/>` child element on the asset.
- **Spine layout:** one `<asset-clip>` per real clip. `offset` = cumulative timeline position. `start` = source in-point (with the NTSC nudge). Gaps emit `<gap>` elements when `include_gaps=True`.
- **Asset reuse:** one `<asset>` per *source* file (not per clip), IDs assigned in discovery order as `r2`, `r3`, etc.
- **Timeline name from filename:** `title=` defaults to None; when None, derives from the output filename's basename.
- **Group filter:** `export_fcpxml(... group_filter=...)` is applied in all three internal clip-loops (used-source discovery, sequence-duration calc, spine emission). Audio summary in the export dialog passes the same filter through `get_export_audio_summary` so the line reflects only the filtered sources.

### OTIO Export (OpenTimelineIO native JSON)

`core.otio_exporter.export_otio` writes an `.otio` document matching the public OTIO schema (Timeline.1 / Stack.1 / Track.1 / Clip.2 / Gap.1 / ExternalReference.1 / TimeRange.1 / RationalTime.1). Resolve Studio imports it natively; Premiere goes through an external OTIO adapter.

- **No `opentimelineio` Python dep:** we emit the JSON directly. OTIO 0.18.1 has a packaging bug on Python 3.14, and we only ever wrote files.
- **`source_range.start_time` nudge (`_SRC_SEEK_NUDGE = 0.25`):** every clip's `source_range.start_time.value` is written as `frame + 0.25`. *Why:* at rates like 24000/1001, the round-trip `time = value / our_rate; frame = floor(time * reader_rate)` drifts by ±1 ULP. A `+0.25` nudge eliminates all drift under `floor()`, `trunc()`, and `round()` reader modes. FCPXML uses `+0.5` (different rounding mechanism); OTIO needs `+0.25` because `+0.5` breaks readers using `round()`. Only `source_range.start_time` is nudged.
- **Clip.2 schema:** `media_references` dict + `active_media_reference_key="DEFAULT_MEDIA"`.
- **`available_range`** on each `ExternalReference` = `(start=0, duration=source.total_frames)` at sequence rate.
- **Asset reuse:** one `ExternalReference` *dict* per source file shared by dict identity across clips.
- **Metadata round-trip:** `metadata["prismasynth"]` stores `source_id` / `clip_id` / `color_index` / `label` / `group_ids` so a future OTIO importer could reconstruct a `.psynth` project.
- **Group filter:** same plumbing as the FCPXML exporter.

### Thumbnail System

Persistent, long-lived thumbnail generator with sweep-based sequential decode. Created once per session, survives timeline edits. CPU-only decode (no NVDEC) to avoid GPU contention with mpv scrubbing.

- **Visible-only scope:** Only generates thumbnails for clips currently in the timeline viewport.
- **Playhead sweep:** From the playhead, sweeps right (forward) then left (backward) through sorted frame targets. Within each sweep, frames within `_SEQUENTIAL_THRESHOLD` (400 frames) are decoded sequentially via PyAV (`_grab_frames_sweep`) — avoids redundant re-seeks from the same keyframe. Distant frames fall back to independent ffmpeg seeks (`_grab_frame_single`).
- **LQ proxy placeholders:** If a `.proxy` file exists (from scene detection), instantly upscales 48x27 frames to 192x108 as blurry placeholders while HQ thumbnails load.
- **6 parallel workers:** Persistent `ThreadPoolExecutor`. Sweep runs and single-frame seeks are both submitted to the pool.
- **Persistent coordinator thread:** Runs for the session. `notify_clips_changed()` rebuilds the clip lookup without destroying the cache or thread pool.
- **Pause-on-pan, NOT pause-on-scrub:** Middle-mouse pan of the timeline pauses the coordinator (rapid viewport reprioritization is wasteful) — `pan_started`/`pan_ended` signals on `TimelineStrip` connect to `_thumbnail_cache.pause()`/`resume()`. Scrubbing the playhead does NOT pause the coordinator (experiment kept since v0.10.x — with baked thumbnails on disk the coordinator's work is cheap JPEG decode, doesn't contend with mpv).
- **Memory-only cache:** `_mem_cache` dict of QImage at 192x108, keyed by `{source_id}_{frame_num}`. Persists across clip changes (splits, deletes). `.copy()` on QImage to prevent use-after-free across thread boundaries.
- **GPU isolation:** Thumbnails use CPU-only ffmpeg (no `-hwaccel cuda`) to leave NVDEC exclusively for mpv scrubbing.
- **Cache Thumbnails dialog:** `ui.cache_thumbnails_dialog.CacheThumbnailsDialog` is a user-triggered modal that bakes the on-disk thumbnail cache for the first AND last frame of every non-gap clip on the timeline (or only clips inside the in/out range when set). Backed by `thumbnail_cache.BulkCacheJob`. **Per-source worker cap:** light-pool worker count is bounded by `min(_BULK_LIGHT_WORKERS, max(1, unique_light_sources * 2))` — fixes the historical "Fantasy Island" PyAV native segfault that triggered when 4+ workers had containers open on the same source. Single-source bakes drop from 8 → 2 workers (~2-3× slower); multi-source bakes (≥4 unique sources) are unaffected.

### Proxy System (Scene Detection Only)

`ProxyFile` (48x27 mmap binary) is generated during scene detection for TransNetV2 input reuse. Stored at `%LOCALAPPDATA%/prismasynth/cache/proxies/`. Managed by `ProxyManager`. Also used as LQ thumbnail placeholders. Supports `frame_offset` (stored in `.offset` sidecar file) for proxies that don't start at source frame 0.

### Import & Scene Detection (separate steps)

1. **Import** (`ImportDialog` or drag-and-drop) — multi-file select, probes all files, creates one whole-file clip per source appended sequentially. All files in a batch must share the same resolution and FPS.
2. **Detect Cuts** (`DetectDialog`, Ctrl+D by default) — runs SceneDetector on **all non-gap clips** on the timeline. Two detector backends are exposed via a dropdown:
   - **TransNetV2** (default) — fast, hard-cut neural network. Per-segment loop. HSV differencing as automatic fallback if TransNetV2 fails to load.
   - **OmniShotCut** — slower transformer alternative. Filtered to hard cuts only (`inter_label == 1`); soft transitions are skipped per user preference.
   Scene detection decode fallback chain (shared by both detectors via `core.ffmpeg_decode.decode_to_array`): (1) parallel 4-segment ffmpeg with `scale_cuda` (NVDEC + GPU resize), (2) parallel 4-segment ffmpeg with CPU scale, (3) single ffmpeg CPU fallback. mpv and thumbnails are paused during detection to avoid GPU contention.

### OmniShotCut Sidecar Architecture

OmniShotCut requires `torch==2.5.1` / Python 3.10 / CUDA 12.4 — incompatible with PrismaSynth's Python 3.13+ free-threaded + `torch cu126`. So OmniShotCut runs in a separate venv (`venv-omnishotcut/` at repo root, gitignored) as a sidecar subprocess.

- **Setup** — `scripts/setup_omnishotcut.py` installs Python 3.10 via `uv`, creates the sidecar venv, downloads the model checkpoint to `%LOCALAPPDATA%/prismasynth/models/OmniShotCut_ckpt.pth`, writes a `venv-omnishotcut/.prismasynth_ready` sentinel as the LAST step. Re-runnable with `--repair`.
- **Sidecar** — `scripts/omnishotcut_sidecar.py`. Loads the model ONCE per Detect Cuts run (5-15s) and processes all segments sequentially. Communicates over JSON-line stdio + raw frame bytes.
- **Runner** — `core.omnishotcut_runner.OmnishotcutRunner`. Spawns the sidecar with `CREATE_NEW_PROCESS_GROUP` (Windows) so cancellation can send `CTRL_BREAK_EVENT` for graceful shutdown before `proc.kill()`.
- **Setup detection** — `is_setup_complete()` checks all three of: sentinel file, sidecar python.exe, and `third_party/OmniShotCut/` exist. The sentinel is written LAST so a partially-installed venv is correctly reported as not-ready.

### Project Portability & Relinking

`.psynth` files store both an absolute `file_path` and a `relative_path` (relative to the project file's directory) per source. On load, if the absolute path is missing, `core.project.load_project()` silently falls back to the relative path resolved against the project file's directory — sources that travel together with the project (e.g. zipped to a USB stick) reopen with no UI prompt.

For sources that still don't resolve, `_load_from()` shows `ui.relink_dialog.RelinkDialog` BEFORE clearing the previously-open project's state — cancelling the dialog leaves the previous project intact (no destructive load). The dialog uses folder pickers per row.

`save_project()` wraps `os.path.relpath` in `try/except ValueError` to handle Windows cross-drive paths (stored as `relative_path: null`). On load, `pixels_per_frame` (zoom level) is restored BEFORE `scroll_offset` so the saved pixel offset is interpreted at the right scale.

### Source Timebase Audit & Auto-fix

PrismaSynth detects when a source's container `time_base` can't exactly represent its declared fps and surfaces a one-click ffmpeg remedy. *Why:* at NTSC 23.976 (1001/24000 s per frame), a container timebase of `1/16000` averages 667.333 ticks per frame; FCPXML / OTIO imports into NLEs that time-seek the source (Resolve, Premiere) drift by ±1 frame because the rounded tick lands inside the wrong source frame. The exporter's existing `_src_seek_str` `+frame_num/2` nudge is verified only against PrismaSynth-exported MOVs (`time_base=1/24000000`); it can't protect against an inexact source-side tick grid because Resolve's seek lands inside the source's own (broken) ticks before the nudge can help.

- `core.video_source.is_seek_safe(time_base_num, time_base_den, fps)` — module-level helper. Test: `(frame_num * time_base_den) % (frame_den * time_base_num) == 0` where `(frame_num, frame_den) = _frame_duration_for_fps(fps)`. Unknown timebase (`0/0`) returns True so legacy projects never trip the warning.
- `utils/ffprobe.py::probe_video` reads `stream.time_base` and exposes `time_base_num/den` on `VideoInfo`. Persisted on `VideoSource`; round-trips through `.psynth`. Legacy projects re-probe on first load alongside the existing audio reprobe.
- `MainWindow._warn_unsafe_timebases` fires from import, drag-drop, project load, and relink. Lists unsafe sources via `ui.timebase_warning_dialog.TimebaseWarningDialog` with per-row Input / Output paths + sibling-`.mkv` detection.
- Auto-fix path: `TimebaseWarningDialog → RemuxProgressDialog → core.timebase_remuxer.RemuxJob` (QThread; sequential ffmpeg with cancel-kills-proc + per-source `source_done` signals). The command shape is `ffmpeg -y -nostdin -v error -progress pipe:1 -i <input> -map 0:v:0 -map 0:a:0? -c:v copy -video_track_timescale <N> <audio-tail> <output>`. Video is always stream-copied; only the first audio stream is mapped (matches what `core.exporter` does — discards Bluray-rip commentary / dubs). MainWindow probes each fixed file and routes through the existing `_apply_relink_results` — same machinery as Tools → Relink…, so reader/proxy/thumbnail/clip-clamp logic isn't duplicated.
- Audio mode (chosen via global `QComboBox` in the warning dialog, persisted under QSettings key `timebase_autofix/audio_mode`):
  - `keep` → `-c:a copy` (default; preserves original codec + layout)
  - `reencode_same` → `-c:a pcm_s16le -ar 48000` (PCM 16-bit @ 48 kHz, layout preserved — 5.1 stays 5.1)
  - `stereo` → `-c:a pcm_s16le -ar 48000 -ac 2` (PCM 2.0; ffmpeg's default downmix matrix condenses surround channels into L/R rather than dropping them — no audio is lost. Mono is upmixed)
- Per-source progress: `RemuxJob` parses `out_time_us=` lines from ffmpeg's `-progress pipe:1` stdout in the foreground while a daemon thread drains stderr (mirrors `core.exporter._drain` to avoid the OS-pipe-fills-and-deadlocks failure). Throttled `source_progress(source_id, fraction)` emits at ≥1% advancement; the dialog combines this with the outer `(idx, total)` step counter into a single 0-1000-resolution bar reading `Remuxing X of N (P%): <name>`.
- Output path: `<basename>_fixed.mov` next to the original, MOV regardless of input container (Resolve compatibility — that's why `.mov` was the original target). Stream copy preserves frame count + ordering, so `Clip.source_in/source_out` indices line up 1:1 in the fixed file — all timeline edits transfer for free since `Clip.source_id` doesn't change during relink.
- Sibling `<basename>.mkv` is auto-detected and preferred as the remux input — avoids double-remuxing through the broken-timebase intermediary. Implemented in `ui.timebase_warning_dialog._pick_remux_input`.
- Target timescale = `frame_den` returned by `_frame_duration_for_fps(fps)` — matches the FCPXML denominator the exporter writes (24000 for 23.976, 30000 for 29.97, F·100 for integer F). Verified safe by `is_seek_safe` for every standard rate.
- `ui/source_info_dialog.py` renders a Timebase row: grey "1/X (exact)" when safe, yellow "1/X — drift risk for FCPXML / OTIO" with tooltip when unsafe.

### Threading Model

| Thread | What | Key constraint |
|--------|------|---------------|
| Main (Qt) | All UI, signal slots, mpv commands | QPixmap only here |
| Thumbnail coordinator | Persistent thread + 6-worker pool, sweep decode via PyAV + ffmpeg | Pauses during middle-mouse pan (not scrub), playhead-distance sweep |
| Scene detection (QThread) | Parallel ffmpeg decode + TransNetV2 inference (OR drives OmniShotCut sidecar) | Stores subprocess refs for cancellation |
| OmniShotCut sidecar | Separate Python 3.10 process; loads model once, infers per segment | JSON-line stdio + raw bytes; cancel via CTRL_BREAK_EVENT then kill |
| Export | Per-segment ffmpeg subprocesses (ThreadPoolExecutor) | Signals must use QueuedConnection to UI |

**Thread safety:** `threading.Lock` on VideoReader, FrameCache, and export process lists. Qt signals for cross-thread communication. ThumbnailCache emits `QImage` (thread-safe) not `QPixmap` (main-thread-only). Exporter uses `QueuedConnection` for all UI signal connections. Do NOT use `faulthandler.enable()` on Windows — it intercepts mpv's internal structured exceptions (`0xe24c4a02`) and kills the process.

### Gaps

W deletes a clip, leaving a gap (same-duration `Clip` with `source_id=None`). D ripple-deletes (collapses space). Adjacent gaps auto-merge. Gaps are selectable, visible as dark dashed rectangles. Skipped during export by default; with **Include gaps** checked in the export dialog, they render as black + silence (Video / Image Seq / Audio Only) or `<gap>` markers (XML / OTIO). Playable during playback (shows black, advances at source FPS). Deleting a gap (W or D) teleports the playhead to the gap's position.

### Timeline Widget

Custom-painted strip (not Qt model/view). Clip positions computed from cumulative frame counts via `_frame_to_pixel()` / `_pixel_to_frame()` to prevent truncation drift. Both account for `TIMELINE_H_PADDING`. Playhead clamped to content area. Track height is draggable (30-200px). Thumbnails scale to fill track height at 16:9 aspect with 6px color border. Marquee selection by dragging below the track (Ctrl to add). Timeline widget wrapped in a container with left/right margins.

**Paint order:** background → ruler → clips → group label strip below each clip → in/out overlay → cut preview line → playhead → marquee rectangle.

**Pan/scrub signals:** `scrub_started/ended` signals fire on left-click playhead drag (used to pause the mpv preview); `pan_started/ended` signals fire on middle-mouse drag (used to pause the thumbnail coordinator).

## Key Conventions

- Keyboard shortcuts go through `core/shortcuts.py::ShortcutManager`. NEVER call `QAction.setShortcut(...)` directly — use `self._shortcut_mgr.attach_action(sid, action)` so user overrides are respected.
- `TimelineModel.add_clips(assign_colors=False)` when loading from project file to preserve saved colors.
- `self._sources` dict is updated in-place (`clear()` + `update()`) never reassigned, because other objects hold references to it.
- Exporter stored as `self._exporter` on MainWindow to prevent garbage collection during background export.
- `_playback_updating` flag in MainWindow distinguishes playback-driven playhead updates from user scrubs.
- Never show downsampled proxy frames in the preview — full-quality scrubbing via mpv at all times.
- On Windows, mmap'd proxy files cannot be overwritten — `ProxyFile.save_frames` catches `OSError` and skips gracefully. The Cache Manager calls `proxy_manager.close_all()` before clearing the proxies category.
- `TimelineModel.clear()` also clears the undo stack, in/out points, **and groups** (fresh start for new/load project).
- Mutation methods that modify `_clips` or `_groups` call `_push_undo()` after validation passes, before the actual mutation. Selection-only changes and in/out point changes are not undoable.
- `_on_new_project()` must stop the thumbnail cache before clearing readers/sources to avoid deadlocks.
- `load_source()` calls `_ensure_video_visible()` first — without this, `wait_for_property('seekable')` hangs if `vid='no'` was set by `clear_frame()`.
- Pause mpv and thumbnail cache before scene detection to prevent GPU contention crashes.
- All FFmpeg commands should include `-nostdin` (prevents stdin reads on Windows) and `-v error` (consistent error capture).
- Export uses `-frames:v N` (not `-t`) for frame-accurate output. Pre-input `-ss` timestamps go through `Exporter._frame_to_seek_ts()` (half-frame margin).
- Export temp-file extension matches the output extension (`.mov` for ProRes, `.mp4` for H.264/5, `.mkv` for FFV1). Hardcoding `.mkv` would silently drop muxer options like `-video_track_timescale` on non-Matroska outputs.
- Export subprocesses MUST be registered via `Exporter._register_proc()`. Never write to `self._active_procs` directly.
- `ExportDialog` cancel button is state-driven: "Cancel" (idle) → "Cancel Export" (running) → "Close" (done).
- All five exports share `ExportDialog`. Don't reintroduce per-format dialogs — settings dict carries `mode` ("video" / "image_sequence" / "xml" / "otio") plus shared keys (`include_gaps`, `use_render_range`, `group_filter`).
- FCPXML asset-clip `start` uses `_src_seek_str` with a `+frame_num/2` numerator nudge. OTIO `source_range.start_time.value` uses `+_SRC_SEEK_NUDGE (0.25)`. **Different root causes, different nudge values — don't cross-port. Don't revert either to unnudged seek positions.**
- `TimelineModel.set_in_point()` / `set_out_point()` reject invalid inputs (would make In >= Out) instead of silently wiping the opposite marker — except for the collision-nudge case described in the In/Out section above.
- `TimelineModel.add_clips([])` and `replace_detected({})` early-return without pushing undo (would clobber the redo stack on empty ops like a failed import).
- `save_project()` writes to `filepath + ".tmp"` then `os.replace()` — atomic, so the 60s autosave can't destroy the project mid-write.
- `PreviewWidget.load_source()` passes `timeout=5.0` to `wait_for_property('seekable')` — without a timeout, a corrupt or unreachable file would hang the UI thread.
- Thumbnail subprocess must use `communicate()` (not `stdout.read()` + `wait()`) to prevent Windows handle leaks.
- `run.bat` sets `PYTHON_GIL=1` for Python 3.13+ free-threaded builds — required for correct threading behavior.
- `playback_engine.py` exists in `src/core/` but is unused — mpv handles playback natively. Retained as reference only.
- Autosave fires every 60 seconds when the project is dirty (has unsaved changes).
- `TimelineStrip` emits `scrub_started` / `scrub_ended` signals on playhead drag and `pan_started` / `pan_ended` on middle-mouse pan. Pan signals connect to thumbnail-cache pause/resume; scrub signals connect to mpv preview pause.
- `TimelineModel.replace_detected(replacements)` swaps clips by ID with detected sub-clip lists — the core integration point after scene detection. Preserves gaps and non-matched clips.
- `ThumbnailCache` is created once and lives for the session. `_start_thumbnail_cache()` creates it on first call, subsequent calls use `notify_clips_changed()` + `reprioritize()`. Only `_on_new_project()` destroys it.
- `ProxyManager.load_or_open(source, force_reopen=True)` is used after detection to pick up newly saved proxy files.
- `_load_from()` runs the relink dialog BEFORE clearing state. Do not reorder — cancelling the dialog must leave the previous project intact.
- `save_project()` writes `relative_path` per source (relative to the `.psynth` directory) alongside the absolute path. Cross-drive paths on Windows (`os.path.relpath` raises `ValueError`) get stored as `relative_path: null`.
- `RelinkDialog` per-row Browse uses `QFileDialog.getExistingDirectory`. Don't switch back to file pickers; the folder UX is what users wanted.
- FCPXML / OTIO `export_*(... title=None ...)` derive the title from `output_path`. Don't restore the old `title="PrismaSynth"` literal default.
- OmniShotCut runs in **hard-cuts-only mode** by default. To re-enable the full transition set, send `hard_cuts_only: false` in the per-segment JSON header — there is currently no UI toggle for this.
- OmniShotCut sidecar is launched ONCE per Detect Cuts run, not per segment — model load is 5-15s and must be amortised. Do not refactor to per-segment subprocess spawning.
- `core.omnishotcut_runner.is_setup_complete()` checks the sentinel file (`venv-omnishotcut/.prismasynth_ready`) AND the venv python AND `third_party/OmniShotCut/`. The sentinel is written as the LAST step of `scripts/setup_omnishotcut.py`.
- The decode helpers in `core.ffmpeg_decode` are SHARED by both detector backends. Don't fork them per detector.
- OmniShotCut has no LICENSE file in its upstream repo — vendored snapshot at `third_party/OmniShotCut/` is gitignored. Personal-use only until upstream clarifies licensing.
- `decode_to_array()` returns `(padded_array, n_frames, method, elapsed_secs)`. The padded array's leading and trailing slack is left UNINITIALIZED — caller must zero or pad-replicate it if the detector requires deterministic edges (e.g. TransNetV2 replicates first/last frame).
- `core/group.clip_matches_filter(clip, group_filter)` is the single source of truth for the export-side group filter. Used by `Exporter._build_segments`, `xml_exporter.export_fcpxml`, `otio_exporter.export_otio`, `TimelineModel.compute_export_extent`, and `TimelineModel.get_used_source_ids`. None = no filter.
- The `(Untagged)` row in the group filter applies only to non-gap clips. Gaps are governed by **Include gaps** alone — the filter never drops gap segments.
- Bulk thumbnail cache: `_BULK_LIGHT_WORKERS` is the **upper bound**, not a fixed value. The actual concurrency is `min(upper_bound, max(1, unique_light_sources * 2))` to avoid the PyAV native segfault that triggers on 4+ concurrent containers per source.
- `is_seek_safe` returns True for unknown timebase (`time_base_num/den == 0`). Don't change this default — legacy projects haven't been re-probed yet, and a False default would pop a spurious warning on every load until the reprobe lands.
- The fixed remux output is **always `.mov`** even when the sibling `.mkv` is used as input. The user's whole motivation for the original `.mov` was Resolve compatibility; emitting `.mkv` would force them to re-remux a second time. See `ui/timebase_warning_dialog._fixed_output_path`.
- `MainWindow._take_pre_op_snapshot()` flushes the project to disk BEFORE copying it into the versions store. So the snapshot reflects the working state in front of the user — not whatever the last autosave wrote. Mirror this in any new pre-op call site.
- `MainWindow._load_from(path, project_path_override=...)` is the version-restore entry point: load `path`'s contents in-memory while keeping `self._project_path` pointed at `project_path_override`. After restore, MainWindow marks dirty so the restored state writes through on the next autosave. Don't reuse the override param for anything else — it's specifically tied to the restore flow.
- `ProjectVersionStore.create()` reads the manifest with `_read_manifest()` directly, NOT `list_versions()`. Reason: `list_versions()` self-heals orphan `.psynth` files into the manifest, and the file we just copied IS on disk but NOT yet in the manifest — that path would re-derive it as an orphan, then `manifest.insert(0, entry)` dupes it.
- File-dialog directory defaults live in QSettings under `dialogs/`: `dialogs/last_project_dir` (read+written by Open Project, written by Save Project As — Save As only reads it when no project is loaded, otherwise it defaults to the loaded project's directory) and `dialogs/last_import_dir` (read by `_on_import` when constructing `ImportDialog`, written by `_on_import_complete` so the same memory captures dialog Browse, drag-drop, AND Media Panel drops). ExportDialog takes a `default_dir=` kwarg from MainWindow — computed as the project's directory when saved, else the first non-gap clip's source directory — and reuses it across all six Browse buttons (Video / audio sidecar / image folder / audio-only / XML / OTIO). No per-Browse memory inside the export dialog; the rule is deterministic by spec.
