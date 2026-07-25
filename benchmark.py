"""Benchmark the Nelux ComfyUI nodes against ComfyUI's built-in video path.

ComfyUI's stock video nodes are backed by PyAV: ``VideoFromFile.get_components``
decodes every frame through PyAV into an rgb24 ndarray, and
``VideoFromComponents.save_to`` encodes with PyAV's software H.264 encoder. The
``comfy_*`` functions below reimplement exactly those code paths so the two
sides can be timed in one process, on one machine, against one file.

Usage:

    python benchmark.py CLIP.mp4 [--frames 300] [--repeats 3] [--json out.json]
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _load_nodes():
    path = Path(__file__).resolve().parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("comfyui_nelux_nodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PeakRSS:
    """Samples this process's resident set size while a block runs.

    Only meaningful in a process that does nothing else -- RSS is not returned to
    the OS promptly, so a peak measured after other work has already run mostly
    reports the high-water mark of that earlier work. The memory section below
    therefore re-execs this script per case (see --mem-case)."""

    def __init__(self, interval=0.005):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak = 0

    def _run(self):
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def measure(fn, repeats=3, warmup=True):
    """Return (median seconds, last result)."""
    if warmup:
        fn()
    timings, result = [], None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


# --------------------------------------------------------------------------- #
# ComfyUI's built-in path, reimplemented (comfy_api VideoFromFile /
# VideoFromComponents, which are PyAV-backed).
# --------------------------------------------------------------------------- #
def comfy_probe(path):
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        return {
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
            "total_frames": stream.frames,
            "duration": float(container.duration or 0) / 1_000_000,
        }


def comfy_decode(path, limit=None):
    """VideoFromFile.get_components(): every frame to rgb24, then float32 [0,1]."""
    import av

    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if limit and len(frames) >= limit:
                break
    return torch.stack(frames).float() / 255.0


def comfy_encode(images, fps, out_path, crf=20, preset="medium"):
    """VideoFromComponents.save_to(): PyAV software H.264."""
    import av

    with av.open(out_path, mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[2]
        stream.height = images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        for image in images:
            array = (image * 255).clamp(0, 255).byte().cpu().numpy()
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out_path


def comfy_transcode(src, out_path, limit=None, crf=20, preset="medium"):
    """What a stock Load Video -> Save Video does: decode every frame into RAM as
    float32, then encode the batch."""
    images = comfy_decode(src, limit)
    fps = comfy_probe(src)["fps"]
    return comfy_encode(images, fps, out_path, crf=crf, preset=preset)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_time(seconds):
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, label, seconds, frames=None, size=None, baseline=None):
        self.rows.append({
            "group": group, "label": label, "seconds": seconds,
            "fps": (frames / seconds) if frames else None,
            "output_bytes": size,
            "baseline": baseline,
        })
        row = self.rows[-1]
        speedup = f"  {baseline / seconds:5.2f}x" if baseline is not None else ""
        fps = f"  {row['fps']:8.1f} fps" if row["fps"] else ""
        mib = f"  {size / 2**20:6.2f} MiB" if size else ""
        print(f"  {label:34s} {_fmt_time(seconds):>9s}{fps}{mib}{speedup}")
        return seconds

    def add_memory(self, label, peak_bytes, baseline=None):
        self.rows.append({
            "group": "memory", "label": label,
            "peak_rss_mb": peak_bytes / 2**20, "baseline": baseline,
        })
        ratio = f"  {baseline / peak_bytes:5.2f}x" if baseline else ""
        print(f"  {label:34s} {peak_bytes / 2**20:8.0f} MB{ratio}")
        return peak_bytes


# --------------------------------------------------------------------------- #
# Peak memory
#
# The architectural claim is that a stock Load Video -> Save Video materializes
# every frame as float32 (W x H x 12 bytes each) while the Nelux path streams one
# frame at a time. Measuring that needs a clean process per case: RSS is not
# handed back to the OS promptly, so a peak sampled after other work has run
# mostly reports the earlier work's high-water mark.
# --------------------------------------------------------------------------- #
def _run_memory_case(case, clip, frames, out_path):
    """Executed in a fresh subprocess; prints one JSON line with the peak RSS."""
    with _PeakRSS() as rss:
        if case == "comfy":
            comfy_transcode(clip, out_path, frames)
        else:
            nodes = _load_nodes()
            accelerator, codec = case.split(":", 1)
            fps = float(nodes._probe(clip)["fps"])
            nodes._transcode_nelux(
                clip, out_path, codec=codec, preset="p4", cq=20,
                decode_accelerator=accelerator, duration=frames / fps, audio=False,
            )
    print("PEAK_RSS " + json.dumps({"case": case, "peak_rss": rss.peak}))


def _measure_memory(case, clip, frames, out_path):
    import subprocess

    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), clip, "--frames", str(frames),
         "--mem-case", case, "--mem-out", out_path],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("PEAK_RSS "):
            return json.loads(line[len("PEAK_RSS "):])["peak_rss"]
    raise RuntimeError(f"memory case {case} failed:\n{result.stdout}\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mem-frames", type=int, default=200,
                        help="frames used for the peak-memory comparison")
    parser.add_argument("--json")
    parser.add_argument("--mem-case", help=argparse.SUPPRESS)
    parser.add_argument("--mem-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mem_case:
        _run_memory_case(args.mem_case, str(Path(args.clip).resolve()),
                         args.frames, args.mem_out)
        return 0

    nodes = _load_nodes()
    nelux = nodes._import_nelux()
    clip = str(Path(args.clip).resolve())
    props = nodes._probe(clip)
    n = min(args.frames, int(props["total_frames"]) or args.frames)
    tmp = Path(tempfile.mkdtemp(prefix="nelux-bench-"))

    accelerators = ["cpu"]
    if nodes._accelerator_works(nelux, clip, "nvdec"):
        accelerators.append("nvdec")

    print(f"\nclip      {clip}")
    print(f"          {props['width']}x{props['height']} {props['fps']:g} fps "
          f"{props['codec']} {props['pixel_format']}, {props['total_frames']} frames")
    print(f"frames    {n}   repeats {args.repeats}   accelerators {accelerators}")
    print(f"nelux     {getattr(nelux, '__version__', '?')}   torch {torch.__version__}")

    report = Report()

    # -- metadata ----------------------------------------------------------- #
    print("\nMetadata (header only, no decode)")
    base, _ = measure(lambda: comfy_probe(clip), args.repeats)
    report.add("probe", "ComfyUI (PyAV av.open)", base)
    seconds, _ = measure(lambda: nodes._probe(clip), args.repeats)
    report.add("probe", "nelux.probe", seconds, baseline=base)

    # -- decode to an IMAGE batch ------------------------------------------- #
    print(f"\nDecode {n} frames to an IMAGE batch")
    base, _ = measure(lambda: comfy_decode(clip, n), args.repeats)
    report.add("decode", "ComfyUI (PyAV)", base, frames=n)
    load = nodes.NeluxLoadFrames()
    for accelerator in accelerators:
        seconds, _ = measure(
            lambda a=accelerator: load.load(clip, 0, n, 1, a, "cpu", 0, 0),
            args.repeats,
        )
        report.add("decode", f"Nelux Load Frames [{accelerator}]", seconds,
                   frames=n, baseline=base)

    # -- encode an IMAGE batch ---------------------------------------------- #
    print(f"\nEncode {n} frames from an IMAGE batch")
    images = load.load(clip, 0, n, 1, accelerators[-1], "cpu", 0, 0)[0]
    fps = float(props["fps"])
    base, _ = measure(
        lambda: comfy_encode(images, fps, str(tmp / "comfy.mp4")), args.repeats)
    report.add("encode", "ComfyUI (PyAV libx264)", base, frames=n,
               size=os.path.getsize(tmp / "comfy.mp4"))
    for codec in ("libx264", "h264_nvenc", "hevc_nvenc"):
        out = tmp / f"nelux_{codec}.mp4"
        try:
            seconds, _ = measure(
                lambda c=codec, o=out: nodes._nelux_encode(
                    images, fps, str(o), codec=c, preset="p4", cq=20),
                args.repeats,
            )
        except Exception as exc:
            print(f"  {f'Nelux Encode Frames [{codec}]':34s}  unavailable ({exc})")
            continue
        report.add("encode", f"Nelux Encode Frames [{codec}]", seconds, frames=n,
                   size=os.path.getsize(out), baseline=base)

    # -- full transcode ----------------------------------------------------- #
    print(f"\nTranscode {n} frames (Load Video -> Save Video)")
    base, _ = measure(
        lambda: comfy_transcode(clip, str(tmp / "comfy_tc.mp4"), n), args.repeats)
    report.add("transcode", "ComfyUI (PyAV decode + encode)", base, frames=n,
               size=os.path.getsize(tmp / "comfy_tc.mp4"))
    duration = n / fps
    for accelerator in accelerators:
        for codec in ("libx264", "h264_nvenc"):
            out = tmp / f"nelux_tc_{accelerator}_{codec}.mp4"
            try:
                seconds, _ = measure(
                    lambda a=accelerator, c=codec, o=out: nodes._transcode_nelux(
                        clip, str(o), codec=c, preset="p4", cq=20,
                        decode_accelerator=a, duration=duration, audio=False),
                    args.repeats,
                )
            except Exception as exc:
                print(f"  {f'Nelux [{accelerator} -> {codec}]':34s}  unavailable ({exc})")
                continue
            report.add("transcode", f"Nelux [{accelerator} -> {codec}]", seconds,
                       frames=n, size=os.path.getsize(out), baseline=base)

    # -- peak host memory --------------------------------------------------- #
    # Capped deliberately: the ComfyUI path holds W x H x 12 bytes per frame, so
    # a whole 1080p clip is tens of GB and would just measure the swap file.
    long_n = min(args.mem_frames, int(props["total_frames"]) or args.mem_frames)
    print(f"\nPeak host RSS transcoding {long_n} frames (fresh process per case)")
    base = report.add_memory(
        "ComfyUI (PyAV decode + encode)",
        _measure_memory("comfy", clip, long_n, str(tmp / "mem_comfy.mp4")),
    )
    for accelerator in accelerators:
        codec = "h264_nvenc" if accelerator == "nvdec" else "libx264"
        report.add_memory(
            f"Nelux [{accelerator} -> {codec}]",
            _measure_memory(f"{accelerator}:{codec}", clip, long_n,
                            str(tmp / f"mem_{accelerator}.mp4")),
            baseline=base,
        )

    if args.json:
        Path(args.json).write_text(json.dumps({
            "clip": clip,
            "properties": {k: props[k] for k in
                           ("width", "height", "fps", "codec", "pixel_format",
                            "total_frames")},
            "frames": n,
            "repeats": args.repeats,
            "nelux": getattr(nelux, "__version__", None),
            "torch": torch.__version__,
            "rows": report.rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
