"""OmniShotCut sidecar — runs inside venv-omnishotcut/ (Python 3.10, torch 2.5.1).

Talks to PrismaSynth's main process via stdin (JSON-line + raw bytes) and stdout
(line-delimited JSON). One process per Detect Cuts run; the model is loaded once
and reused across all segments to amortise the 5-15s load cost.

Protocol:

  Parent → sidecar (stdin):
    Line 1 (JSON):
      {"omnishotcut_repo": "<path>", "checkpoint": "<path>", "num_context_frames": K}
    Then per segment:
      JSON line: {"phase": "segment", "seg_id": "...", "frame_count": N, "fps": F}
      N * process_height * process_width * 3 raw uint8 bytes
    Parent closes stdin → sidecar exits.

  Sidecar → parent (stdout, line-delimited JSON):
    {"phase": "loaded", "vram_mb": N, "process_width": W, "process_height": H,
     "max_window": L}
    {"phase": "analyzing", "seg_id": "...", "frame": F, "total": T}
    {"phase": "result", "seg_id": "...", "ranges": [[s, e], ...]}
    ... (final close happens via stdin EOF, no explicit "done" needed)
    {"phase": "error", "msg": "..."}   on failure (exit non-zero)

CLI:
  python omnishotcut_sidecar.py                  # main run mode
  python omnishotcut_sidecar.py --selftest       # load + run inference on a synthetic clip
      --omnishotcut-repo <path> --checkpoint <path>
"""

import argparse
import json
import os
import sys

# Force UTF-8 on stdout/stderr — Windows cmd.exe defaults to cp1252 which
# can't encode tqdm progress bars or other non-ASCII chars from torch/HF
# loaders. Without this, a single non-ASCII byte in a model-loader log
# line crashes the sidecar mid-detection.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _emit_error(msg):
    _emit({"phase": "error", "msg": msg})
    sys.stdout.flush()
    sys.stderr.write(f"omnishotcut_sidecar error: {msg}\n")
    sys.stderr.flush()


def _import_omnishotcut(omnishotcut_repo: str):
    """Make OmniShotCut's modules importable. Mirrors inference.py's sys.path hack."""
    abs_repo = os.path.abspath(omnishotcut_repo)
    if not os.path.isdir(abs_repo):
        raise FileNotFoundError(f"OmniShotCut repo not found: {abs_repo}")
    if abs_repo not in sys.path:
        sys.path.insert(0, abs_repo)
    # Also chdir there — some upstream code uses os.path.abspath('.') as root
    os.chdir(abs_repo)


def _load_model(checkpoint_path: str):
    """Wraps inference.load_model. Returns (model, model_args, vram_mb)."""
    import torch
    from test_code.inference import load_model

    model, model_args = load_model(checkpoint_path)
    vram_mb = 0
    if torch.cuda.is_available():
        try:
            vram_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            vram_mb = 0
    return model, model_args, vram_mb


def _single_array_inference(video_np, model, model_args, num_context_frames,
                            progress_cb=None):
    """Fork of inference.single_video_inference that takes a pre-decoded numpy
    frame array (T, H, W, 3) uint8 instead of a video file path. Reuses the
    upstream split/prune/merge helpers verbatim.

    progress_cb(frame_done, frame_total) is invoked once per chunk."""
    import torch
    from test_code.inference import (
        split_videos, prune_non_context_ranges, merge_ranges, video_transform,
    )

    max_process_window_length = model_args.max_process_window_length

    pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full = [], [], []

    chunks = split_videos(video_np, max_process_window_length, num_context_frames)
    total_chunks = max(1, len(chunks))
    stride = max(1, max_process_window_length - 2 * num_context_frames)
    total_frames = video_np.shape[0]

    for clip_idx, (chunk, num_pad_frames) in enumerate(chunks):
        if progress_cb is not None:
            progress_cb(min(clip_idx * stride, total_frames), total_frames)

        video_tensor = video_transform(chunk).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            outputs = model(video_tensor)

        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1]
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)

        pred_ranges, pred_intra_labels, pred_inter_labels = [], [], []
        start_frame_idx = 0
        for keep_idx in range(len(query_intra_idx)):
            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())
            end_frame_idx = int(query_range_idx[keep_idx].detach().cpu())
            if start_frame_idx >= end_frame_idx:
                continue
            pred_ranges.append([start_frame_idx, end_frame_idx])
            pred_intra_labels.append(pred_intra_label)
            pred_inter_labels.append(pred_inter_label)
            start_frame_idx = end_frame_idx
            if end_frame_idx >= max_process_window_length - num_pad_frames:
                break

        pred_ranges, pred_intra_labels, pred_inter_labels = prune_non_context_ranges(
            pred_ranges, pred_intra_labels, pred_inter_labels,
            max_process_window_length, num_context_frames,
        )
        pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full = merge_ranges(
            pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full,
            pred_ranges, pred_intra_labels, pred_inter_labels,
        )

    if progress_cb is not None:
        progress_cb(total_frames, total_frames)
    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full


# OmniShotCut inter-label semantics (config/label_correspondence.py).
# inter_labels[i] describes the boundary BEFORE shot i:
#   0 = new_start (chunk artifact, not a real cut)
#   1 = hard_cut
#   2 = transition_source (the source side of a fade/dissolve, etc.)
#   3 = transition (dissolve, fade, wipe, push, slide, zoom, doorway)
#   4 = sudden_jump (rapid jump within otherwise continuous footage)
INTER_HARD_CUT = 1


def _ranges_to_cuts(pred_ranges, inter_labels, total_frames, hard_cuts_only=True):
    """Convert OmniShotCut shot ranges to PrismaSynth cut frame numbers.
    Each range is [start, end_exclusive]; cut between range[i] and range[i+1]
    sits at end[i] - 1 (last frame of shot i). Matches TransNetV2's "last
    frame of each shot" semantics that _cuts_to_clips() shifts by +1.

    When hard_cuts_only=True (default), emit a cut at the end of shot[i] only
    if inter_labels[i+1] == hard_cut; transitions, sudden jumps, and chunk
    new_start markers are skipped. Otherwise emit cuts at every boundary.
    """
    if len(pred_ranges) <= 1:
        return []
    cuts = []
    for i in range(len(pred_ranges) - 1):
        end = int(pred_ranges[i][1]) - 1
        if not (0 <= end < total_frames - 1):
            continue
        if hard_cuts_only:
            next_label = inter_labels[i + 1] if i + 1 < len(inter_labels) else None
            if next_label != INTER_HARD_CUT:
                continue
        cuts.append(end)
    return cuts


def _read_json_line(stream):
    """Read one line from binary stdin and parse as JSON. Returns None on EOF."""
    line = stream.readline()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to parse JSON line {line!r}: {e}")


def _read_exact(stream, n):
    """Read exactly n bytes from binary stdin. Raises EOFError on short read."""
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise EOFError(f"short read: got {len(out)} of {n} bytes")
        out.extend(chunk)
    return bytes(out)


def main_run():
    """Main protocol loop. Reads from stdin, processes segments, exits on EOF."""
    import numpy as np

    stdin = sys.stdin.buffer

    # Phase 1: header
    header = _read_json_line(stdin)
    if header is None:
        _emit_error("EOF before header")
        return 1
    omnishotcut_repo = header["omnishotcut_repo"]
    checkpoint = header["checkpoint"]
    num_context_frames = int(header.get("num_context_frames", 0))

    _import_omnishotcut(omnishotcut_repo)

    # Phase 2: load model
    try:
        model, model_args, vram_mb = _load_model(checkpoint)
    except Exception as e:
        _emit_error(f"model load failed: {e}")
        return 1

    process_height = int(model_args.process_height)
    process_width = int(model_args.process_width)
    max_window = int(model_args.max_process_window_length)

    _emit({
        "phase": "loaded",
        "vram_mb": vram_mb,
        "process_width": process_width,
        "process_height": process_height,
        "max_window": max_window,
    })

    frame_size = process_height * process_width * 3

    # Phase 3: per-segment loop
    while True:
        msg = _read_json_line(stdin)
        if msg is None:
            return 0  # parent closed stdin → clean exit
        if msg.get("phase") != "segment":
            _emit_error(f"unexpected message: {msg!r}")
            return 1
        seg_id = msg["seg_id"]
        frame_count = int(msg["frame_count"])
        fps = float(msg.get("fps", 24.0))

        try:
            raw = _read_exact(stdin, frame_count * frame_size)
        except EOFError as e:
            _emit_error(f"segment {seg_id}: {e}")
            return 1

        video_np = np.frombuffer(raw, dtype=np.uint8).reshape(
            frame_count, process_height, process_width, 3
        )

        def progress(frame_done, frame_total, _seg=seg_id):
            _emit({
                "phase": "analyzing",
                "seg_id": _seg,
                "frame": int(frame_done),
                "total": int(frame_total),
            })

        try:
            pred_ranges, _intra, inter_labels = _single_array_inference(
                video_np, model, model_args, num_context_frames,
                progress_cb=progress,
            )
        except Exception as e:
            _emit_error(f"segment {seg_id} inference failed: {e}")
            return 1

        # Hard-cuts-only is the deliberate behaviour — soft transitions
        # (dissolves, fades, wipes, etc.) are skipped per user request.
        # Toggle via the segment header if a UI option is added later.
        hard_cuts_only = bool(msg.get("hard_cuts_only", True))
        cuts = _ranges_to_cuts(pred_ranges, inter_labels, frame_count,
                               hard_cuts_only=hard_cuts_only)
        _emit({
            "phase": "result",
            "seg_id": seg_id,
            "ranges": [[int(r[0]), int(r[1])] for r in pred_ranges],
            "inter_labels": [int(l) for l in inter_labels],
            "cuts": cuts,
        })


def main_selftest(omnishotcut_repo: str, checkpoint: str):
    """Load the model and run inference on a 60-frame synthetic clip. Exits 0 on success."""
    import numpy as np

    _import_omnishotcut(omnishotcut_repo)

    print(f"[selftest] loading checkpoint: {checkpoint}", flush=True)
    model, model_args, vram_mb = _load_model(checkpoint)
    print(f"[selftest] model loaded — process={model_args.process_width}x{model_args.process_height}, "
          f"max_window={model_args.max_process_window_length}, vram_mb={vram_mb}", flush=True)

    H, W = int(model_args.process_height), int(model_args.process_width)
    n_frames = max(60, int(model_args.max_process_window_length))
    rng = np.random.default_rng(0)
    video_np = rng.integers(0, 256, size=(n_frames, H, W, 3), dtype=np.uint8)

    print(f"[selftest] running inference on {n_frames}x{H}x{W} synthetic clip", flush=True)
    pred_ranges, _, _ = _single_array_inference(
        video_np, model, model_args, num_context_frames=0,
    )
    print(f"[selftest] OK — produced {len(pred_ranges)} shot ranges: {pred_ranges[:5]}{'...' if len(pred_ranges) > 5 else ''}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true",
                        help="Load model and run synthetic inference, then exit.")
    parser.add_argument("--omnishotcut-repo", default=None,
                        help="(selftest only) path to OmniShotCut repo")
    parser.add_argument("--checkpoint", default=None,
                        help="(selftest only) path to checkpoint .pth")
    args = parser.parse_args()

    if args.selftest:
        if not args.omnishotcut_repo or not args.checkpoint:
            print("--selftest requires --omnishotcut-repo and --checkpoint", file=sys.stderr)
            return 2
        try:
            return main_selftest(args.omnishotcut_repo, args.checkpoint)
        except Exception as e:
            print(f"[selftest] FAIL: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    return main_run()


if __name__ == "__main__":
    sys.exit(main())
