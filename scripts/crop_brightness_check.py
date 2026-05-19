"""Brightness regression check for the crop-export pipeline.

The crop exporter historically skipped HDR detection and the tonemap
chain, which meant HDR sources were written as raw PQ-space pixels
treated by players as SDR BT.709 → drastically darker output (Y means
in the single digits / low teens).

This script runs two short ffmpeg commands on the same source: one
mirroring the pre-fix crop pipeline (no HDR handling) and one mirroring
the post-fix pipeline (HDR-aware). It then signalstats both outputs and
reports the per-frame Y-mean delta.

Pass criteria:

* **HDR source**: the fixed pipeline's Y mean must be substantially
  brighter than the unfixed one (delta > 30 on an 8-bit scale).
  Anything less means the tonemap chain isn't activating for this
  source and HDR crops will still come out dark.

* **SDR source**: the fixed pipeline must NOT change brightness vs the
  unfixed one (max |delta| < 5). Anything more means the new chain has
  a regression on SDR sources.

Exit code is 0 on PASS, 1 on FAIL.

Usage::

    venv\\Scripts\\python scripts\\crop_brightness_check.py --video PATH
        [--anchor N]     Start frame (default 0).
        [--crop X,Y,W,H] Sub-rect (default = full frame).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["PATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PATH", "")

from utils.ffprobe import probe_hdr  # noqa: E402


HDR_FILTER = (
    "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,"
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0:peak=10,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


def probe_dims_fps(path: str):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", path],
        capture_output=True, text=True, check=True,
    )
    s = next(s for s in json.loads(r.stdout)["streams"]
             if s.get("codec_type") == "video")
    w, h = int(s["width"]), int(s["height"])
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    fps = num / den if den else 24.0
    return w, h, fps


def run_chain(src: str, out_path: str, anchor: int, fps: float,
              crop_rect, is_hdr: bool, simulate_old: bool):
    """Run an ffmpeg command equivalent to one of the two pipelines.

    ``simulate_old`` skips the HDR chain entirely — matches the
    pre-fix crop_exporter behaviour and lets us reproduce the bug."""
    ss = max(0.0, (anchor - 0.5) / fps)
    pre_ss = max(0.0, ss - 1.0)
    post_ss = max(0.0, ss - pre_ss)
    parts = []
    if is_hdr and not simulate_old:
        parts.append(HDR_FILTER)
    if crop_rect is not None:
        x, y, w, h = crop_rect
        parts.append(f"crop={w}:{h}:{x}:{y}")
    parts += ["fps=16", "setpts=PTS-STARTPTS"]
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
    if pre_ss > 0:
        cmd += ["-ss", f"{pre_ss:.6f}"]
    cmd += ["-i", src]
    if post_ss > 0 and not simulate_old:
        cmd += ["-ss", f"{post_ss:.6f}"]
    cmd += [
        "-an",
        "-vf", ",".join(parts),
        "-frames:v", "81",
        "-fps_mode", "passthrough",
        "-c:v", "prores_aw", "-profile:v", "0", "-pix_fmt", "yuv422p10le",
    ]
    if not simulate_old:
        cmd += ["-colorspace", "bt709", "-color_trc", "bt709",
                "-color_primaries", "bt709"]
    cmd += [out_path]
    subprocess.run(cmd, check=True)


def signalstats_yavg(path: str):
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path,
         "-vf", "signalstats,metadata=print",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return [float(m.group(1)) for m in
            re.finditer(r"YAVG:([0-9.]+)", r.stderr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--crop", default="",
                    help="Sub-rect X,Y,W,H — default = full frame.")
    args = ap.parse_args()

    if not Path(args.video).is_file():
        print(f"FAIL: not a file: {args.video}")
        sys.exit(1)

    w, h, fps = probe_dims_fps(args.video)
    is_hdr = bool(probe_hdr(args.video))
    crop_rect = None
    if args.crop:
        crop_rect = tuple(int(x) for x in args.crop.split(","))
    print(f"Source:   {args.video}")
    print(f"Dims:     {w}x{h} @ {fps:.3f}fps")
    print(f"HDR:      {is_hdr}")
    print(f"Anchor:   {args.anchor}")
    print(f"Crop:     {crop_rect if crop_rect else 'full frame'}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        old_path = os.path.join(tmp, "old.mov")
        new_path = os.path.join(tmp, "new.mov")
        print("Running pre-fix chain (no HDR handling) …")
        run_chain(args.video, old_path, args.anchor, fps,
                  crop_rect, is_hdr, simulate_old=True)
        print("Running post-fix chain (HDR-aware) …")
        run_chain(args.video, new_path, args.anchor, fps,
                  crop_rect, is_hdr, simulate_old=False)
        old_y = signalstats_yavg(old_path)
        new_y = signalstats_yavg(new_path)

    n = min(len(old_y), len(new_y))
    if n == 0:
        print("FAIL: couldn't extract Y means from output.")
        sys.exit(1)
    deltas = [new_y[i] - old_y[i] for i in range(n)]
    abs_d = [abs(d) for d in deltas]
    old_mean = sum(old_y[:n]) / n
    new_mean = sum(new_y[:n]) / n
    max_abs = max(abs_d)
    mean_signed = sum(deltas) / n

    print()
    print(f"Frames compared: {n}")
    print(f"Y mean (pre-fix):  {old_mean:.2f}")
    print(f"Y mean (post-fix): {new_mean:.2f}")
    print(f"Mean signed delta: {mean_signed:+.2f}")
    print(f"Max abs delta:     {max_abs:.2f}")

    if is_hdr:
        # HDR: post-fix should be substantially brighter than pre-fix
        # (the pre-fix wrote raw PQ pixels as SDR → dark output).
        ok = (new_mean - old_mean) > 30.0
        print(f"\nHDR threshold:    new - old > 30   → {'PASS' if ok else 'FAIL'}")
    else:
        # SDR: post-fix should not change brightness materially.
        ok = max_abs < 5.0
        print(f"\nSDR threshold:    max |delta| < 5  → {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
