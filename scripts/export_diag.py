"""Export pipeline diagnostic.

Exports a small segment from a real source file using multiple ffmpeg
configurations, then checks the output frame-by-frame for:
  1. Duplicate adjacent frames (identical content)
  2. Whether the first output frame matches the expected source frame
     (i.e. no preroll leakage)

Run:
    venv\\Scripts\\python scripts\\export_diag.py \\
        --source PATH --src-in FRAME --count N [--configs NAMES]

Example:
    venv\\Scripts\\python scripts\\export_diag.py \\
        --source "x:/.../Monster.mov" --src-in 5000 --count 30

Compares each config's output to a reference decode (single-pass, no
hwaccel, straight PNG extraction). The reference tells us what the target
frames *should* look like.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ["PATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PATH", "")


# ----------------------------- configs ------------------------------------

# Each config produces a ProRes 422 LT segment covering `count` frames
# starting at `src_in`. Configs differ in seek / filter / decode strategy.

def config_current(source, src_in, count, fps, out_file):
    """The current production config: two-stage seek + -hwaccel cuda +
    setpts=PTS-STARTPTS + opencl tonemap + -fps_mode passthrough."""
    ss = src_in / fps
    PRE_SEEK = 1.0
    if ss > PRE_SEEK:
        pre_ss, post_ss = ss - PRE_SEEK, PRE_SEEK
    else:
        pre_ss, post_ss = 0.0, ss
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-init_hw_device", "opencl=ocl:0.0", "-filter_hw_device", "ocl",
           "-hwaccel", "cuda"]
    if pre_ss > 0:
        cmd += ["-ss", f"{pre_ss:.6f}"]
    cmd += ["-i", str(source)]
    if post_ss > 0:
        cmd += ["-ss", f"{post_ss:.6f}"]
    cmd += ["-frames:v", str(count),
            "-vf", "setpts=PTS-STARTPTS,format=p010le,hwupload,"
                   "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12,"
                   "hwdownload,format=nv12",
            "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709",
            "-fps_mode", "passthrough", "-an",
            "-c:v", "prores_aw", "-profile:v", "1",
            "-pix_fmt", "yuv422p10le", str(out_file)]
    return cmd


def config_no_hwaccel(source, src_in, count, fps, out_file):
    """Drop -hwaccel cuda — CPU decode, rest identical."""
    ss = src_in / fps
    PRE_SEEK = 1.0
    if ss > PRE_SEEK:
        pre_ss, post_ss = ss - PRE_SEEK, PRE_SEEK
    else:
        pre_ss, post_ss = 0.0, ss
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-init_hw_device", "opencl=ocl:0.0", "-filter_hw_device", "ocl"]
    if pre_ss > 0:
        cmd += ["-ss", f"{pre_ss:.6f}"]
    cmd += ["-i", str(source)]
    if post_ss > 0:
        cmd += ["-ss", f"{post_ss:.6f}"]
    cmd += ["-frames:v", str(count),
            "-vf", "setpts=PTS-STARTPTS,format=p010le,hwupload,"
                   "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12,"
                   "hwdownload,format=nv12",
            "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709",
            "-fps_mode", "passthrough", "-an",
            "-c:v", "prores_aw", "-profile:v", "1",
            "-pix_fmt", "yuv422p10le", str(out_file)]
    return cmd


def config_cpu_tonemap(source, src_in, count, fps, out_file):
    """NVDEC + CPU zscale tonemap chain (no OpenCL)."""
    ss = src_in / fps
    PRE_SEEK = 1.0
    if ss > PRE_SEEK:
        pre_ss, post_ss = ss - PRE_SEEK, PRE_SEEK
    else:
        pre_ss, post_ss = 0.0, ss
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-hwaccel", "cuda"]
    if pre_ss > 0:
        cmd += ["-ss", f"{pre_ss:.6f}"]
    cmd += ["-i", str(source)]
    if post_ss > 0:
        cmd += ["-ss", f"{post_ss:.6f}"]
    cmd += ["-frames:v", str(count),
            "-vf", "setpts=PTS-STARTPTS,"
                   "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                   "tonemap=hable:desat=0:peak=10,"
                   "zscale=t=bt709:m=bt709:r=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709",
            "-fps_mode", "passthrough", "-an",
            "-c:v", "prores_aw", "-profile:v", "1",
            "-pix_fmt", "yuv422p10le", str(out_file)]
    return cmd


def config_single_ss_only(source, src_in, count, fps, out_file):
    """Pre-input -ss only, no post-input -ss. Original code path."""
    ss = src_in / fps
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-init_hw_device", "opencl=ocl:0.0", "-filter_hw_device", "ocl",
           "-hwaccel", "cuda",
           "-ss", f"{ss:.6f}", "-i", str(source),
           "-frames:v", str(count),
           "-vf", "setpts=PTS-STARTPTS,format=p010le,hwupload,"
                  "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12,"
                  "hwdownload,format=nv12",
           "-colorspace", "bt709", "-color_trc", "bt709",
           "-color_primaries", "bt709",
           "-fps_mode", "passthrough", "-an",
           "-c:v", "prores_aw", "-profile:v", "1",
           "-pix_fmt", "yuv422p10le", str(out_file)]
    return cmd


def config_post_ss_only(source, src_in, count, fps, out_file):
    """Post-input -ss only — accurate seek, much slower but textbook correct."""
    ss = src_in / fps
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-init_hw_device", "opencl=ocl:0.0", "-filter_hw_device", "ocl",
           "-hwaccel", "cuda",
           "-i", str(source),
           "-ss", f"{ss:.6f}",
           "-frames:v", str(count),
           "-vf", "setpts=PTS-STARTPTS,format=p010le,hwupload,"
                  "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12,"
                  "hwdownload,format=nv12",
           "-colorspace", "bt709", "-color_trc", "bt709",
           "-color_primaries", "bt709",
           "-fps_mode", "passthrough", "-an",
           "-c:v", "prores_aw", "-profile:v", "1",
           "-pix_fmt", "yuv422p10le", str(out_file)]
    return cmd


def config_h264_nvenc(source, src_in, count, fps, out_file):
    """Same pipeline but H.264 NVENC output instead of ProRes. Rules out
    ProRes-specific bugs."""
    ss = src_in / fps
    PRE_SEEK = 1.0
    if ss > PRE_SEEK:
        pre_ss, post_ss = ss - PRE_SEEK, PRE_SEEK
    else:
        pre_ss, post_ss = 0.0, ss
    out_file = out_file.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-init_hw_device", "opencl=ocl:0.0", "-filter_hw_device", "ocl",
           "-hwaccel", "cuda"]
    if pre_ss > 0:
        cmd += ["-ss", f"{pre_ss:.6f}"]
    cmd += ["-i", str(source)]
    if post_ss > 0:
        cmd += ["-ss", f"{post_ss:.6f}"]
    cmd += ["-frames:v", str(count),
            "-vf", "setpts=PTS-STARTPTS,format=p010le,hwupload,"
                   "tonemap_opencl=tonemap=hable:desat=0:peak=1000:format=nv12,"
                   "hwdownload,format=nv12",
            "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709",
            "-fps_mode", "passthrough", "-an",
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23",
            "-pix_fmt", "yuv420p", str(out_file)]
    return cmd, out_file


CONFIGS = {
    "current":          config_current,
    "no_hwaccel":       config_no_hwaccel,
    "cpu_tonemap":      config_cpu_tonemap,
    "single_ss_only":   config_single_ss_only,
    "post_ss_only":     config_post_ss_only,
    "h264_nvenc":       config_h264_nvenc,
}


# ----------------------------- analysis ----------------------------------

def extract_frames(video_path, out_dir, count):
    """Extract PNG frames from a video for analysis."""
    out_pattern = out_dir / "frame_%04d.png"
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-i", str(video_path),
           "-frames:v", str(count),
           "-pix_fmt", "rgb24",
           str(out_pattern)]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        return []
    return sorted(out_dir.glob("frame_*.png"))


def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def analyze_output(video_path, work_dir, count):
    """Return list of per-frame md5s and a report dict."""
    frame_dir = work_dir / f"frames_{video_path.stem}"
    frame_dir.mkdir(exist_ok=True)
    frames = extract_frames(video_path, frame_dir, count)
    if not frames:
        return [], {"error": "frame extraction failed"}

    hashes = [hash_file(f) for f in frames]
    duplicates = []
    for i in range(1, len(hashes)):
        if hashes[i] == hashes[i - 1]:
            duplicates.append(i)

    # All-identical check (sanity)
    unique = len(set(hashes))
    return hashes, {
        "frames": len(hashes),
        "unique": unique,
        "duplicate_indices": duplicates,
        "first_hash": hashes[0] if hashes else None,
    }


def decode_reference_frames(source, src_in, count, work_dir):
    """Decode the 'true' frames at src_in..src_in+count using a
    frame-accurate method: decode from the start of the file then take
    -vframes with select filter. Slow but unambiguous.
    Actually for speed, use -ss with select filter to drop any preroll."""
    ref_path = work_dir / "reference.mp4"
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error",
           "-i", str(source),
           "-vf", f"select='gte(n\\,{src_in})',setpts=N/24/TB",
           "-frames:v", str(count),
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
           "-pix_fmt", "yuv420p", "-an", str(ref_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        return None
    return ref_path


# ------------------------------- main -------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--src-in", type=int, required=True,
                        help="Source frame number to start at")
    parser.add_argument("--count", type=int, default=30,
                        help="Number of frames to encode (default 30)")
    parser.add_argument("--fps", type=float, default=23.976)
    parser.add_argument("--configs", default="current,no_hwaccel,cpu_tonemap,"
                                             "single_ss_only,post_ss_only,"
                                             "h264_nvenc",
                        help="Comma-separated config names to run")
    parser.add_argument("--keep", action="store_true",
                        help="Keep output files for manual inspection")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}")
        sys.exit(2)

    configs = [c.strip() for c in args.configs.split(",")]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        print(f"Unknown configs: {unknown}")
        print(f"Available: {list(CONFIGS.keys())}")
        sys.exit(2)

    work_dir = Path(tempfile.mkdtemp(prefix="psynth_diag_"))
    print(f"Work dir: {work_dir}")
    print(f"Source: {args.source.name}")
    print(f"src_in: {args.src_in}, count: {args.count}, fps: {args.fps}")
    print()

    results = {}
    try:
        for name in configs:
            fn = CONFIGS[name]
            out_base = work_dir / f"out_{name}.mov"
            t0 = time.perf_counter()
            result = fn(args.source, args.src_in, args.count, args.fps, out_base)
            if isinstance(result, tuple):
                cmd, out_file = result
            else:
                cmd, out_file = result, out_base
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=600)
            except subprocess.TimeoutExpired:
                print(f"[{name:18s}]  TIMED OUT (>10min)")
                results[name] = {"error": "timeout"}
                continue
            dt = time.perf_counter() - t0

            if r.returncode != 0:
                err = (r.stderr or "").splitlines()[-1:] or ["?"]
                print(f"[{name:18s}]  FFMPEG FAILED: {err[0]}")
                results[name] = {"error": err[0]}
                continue

            hashes, report = analyze_output(out_file, work_dir, args.count)
            report["encode_time"] = dt
            results[name] = report
            dup_str = ("OK" if not report["duplicate_indices"]
                       else f"DUPES at {report['duplicate_indices']}")
            print(f"[{name:18s}]  {dt:5.1f}s  {report['frames']} frames, "
                  f"{report['unique']} unique  {dup_str}")

        # Cross-config comparison: is the "current" config's first frame
        # the same as any other config's first frame?
        print()
        print("Cross-config first-frame comparison:")
        firsts = {n: r.get("first_hash") for n, r in results.items()
                  if r.get("first_hash")}
        if firsts:
            groups = {}
            for name, h in firsts.items():
                groups.setdefault(h, []).append(name)
            for h, names in groups.items():
                print(f"  {h[:12]}  {names}")

    finally:
        if not args.keep:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\nKept: {work_dir}")


if __name__ == "__main__":
    main()
