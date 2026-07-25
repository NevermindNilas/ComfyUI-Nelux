# ComfyUI-Nelux

GPU-accelerated video I/O nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
backed by [Nelux](https://github.com/NevermindNilas/Nelux) — NVDEC decode, NVENC
encode, and lossless audio passthrough.

ComfyUI's built-in video nodes go through PyAV: `get_components()` decodes every
frame in software, and `VideoFromComponents.save_to()` re-encodes with software
H.264. These nodes replace both ends of that pipe, and emit/consume Comfy's real
`VIDEO` type so they drop straight into existing graphs.

**A 1080p transcode runs 11x faster and uses 13x less host RAM** on an RTX 3090.
Full numbers across 720p/1080p/4K, the methodology, and the raw JSON are in
[Benchmarks](#benchmarks).

## Install

### ComfyUI Manager

Search for **Nelux Video Nodes**.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/NevermindNilas/ComfyUI-Nelux
<ComfyUI python> -m pip install nelux
```

The `nelux` wheel is built against one PyTorch minor and refuses to import
against another, so install it with **ComfyUI's own Python**. If ComfyUI Desktop
bundles a torch version with no matching Nelux wheel, the nodes raise a clear
error at load time rather than crashing the process.

On Windows, point the nodes at your FFmpeg DLLs before starting ComfyUI (they are
added to the DLL search path before `nelux` is imported):

```
set NELUX_FFMPEG_DLL_DIR=C:\path\to\ffmpeg\bin
```

## Nodes

All under the **Nelux** category.

| Node | In → Out | Notes |
|------|----------|-------|
| **Nelux Load Video** | file → `VIDEO` | Lazy. Frames decode on demand, on the GPU. |
| **Nelux Save Video** | `VIDEO` → `VIDEO` | Hardware encode + lossless audio passthrough. |
| **Nelux Video Info** | path → dims/fps/frames/codec/… | Container header only, no decode. |
| **Nelux Load Frames** | path → `IMAGE` | Direct frame-range/step decode to an IMAGE batch. |
| **Nelux Encode Frames** | `IMAGE` → path | Direct IMAGE batch → file (no audio). |

## Choosing the engine

Decode and encode each expose their own engine picker, and both default to
`auto`. An explicit choice is always honoured as written — if the hardware
cannot do it, the node raises instead of silently handing back a software
encode you did not ask for.

### Decode — `decode_accelerator`

On **Nelux Load Video** and **Nelux Load Frames**: `auto` · `nvdec` · `qsv` ·
`cpu`. Nelux Save Video reuses whatever the Load Video feeding it was set to.

`auto` probes once per codec for a working hardware decoder and falls back to the
CPU, so the pack works unchanged on a machine with no GPU. The probe opens the
decoder exactly the way the nodes decode (`force_8bit=True`), because that
changes which pixel formats a backend has to handle and so can change the answer.

> **QSV decode is not in nelux 0.16.0** — its `Factory.hpp` maps only CPU and
> NVDEC, so picking `qsv` there raises with a message saying so. The option is
> listed for builds that add it; `auto` simply never selects it. QSV *encode* is
> unrelated and does work — that lives in FFmpeg, not in a nelux decode backend.

### Encode — `encoder_engine`

On **Nelux Save Video** and **Nelux Encode Frames**: `auto` · `nvenc` · `qsv` ·
`cpu`. It pairs with `codec`:

| `codec` | `nvenc` | `qsv` | `cpu` |
|---|---|---|---|
| `h264` (and `auto`) | `h264_nvenc` | `h264_qsv` | `libx264` |
| `hevc` | `hevc_nvenc` | `hevc_qsv` | `libx265` |
| `av1` | `av1_nvenc` | `av1_qsv` | `libsvtav1` |

`auto` walks NVENC → QSV → CPU and takes the first that works. Naming an **exact**
encoder (`h264_nvenc`, `libx265`, …) in `codec` overrides `encoder_engine`
entirely, which also keeps workflows saved before `encoder_engine` existed
loading unchanged.

Availability is decided by **actually opening the encoder**, not by asking FFmpeg
what it was compiled with. `nelux.get_available_encoders()` /
`get_nvenc_encoders()` just enumerate build-time codecs, and this FFmpeg build
lists NVENC, QSV, AMF and VAAPI encoders on every platform — trusting that list
makes `auto` pick `h264_nvenc` on an AMD-only box and then die inside
`avcodec_open2`. Opening the encoder also catches per-GPU limits a name list
never could: `av1_nvenc` is in every build and fails on an RTX 3090, because AV1
encode starts at Ada. Each probe runs once and is cached (~80 ms NVENC, ~500 ms
QSV); software encoders are never probed.

## Benchmarks

Measured with [`benchmark.py`](benchmark.py), which times these nodes against a
faithful reimplementation of ComfyUI's own PyAV code paths (`VideoFromFile.
get_components` and `VideoFromComponents.save_to`) in a single process, so both
sides see the same machine, file and clock. Median of 3 runs after a warm-up.

> **Hardware:** Intel Core i7-13700K, RTX 3090 (driver 610.74), Windows 11,
> Python 3.14, torch 2.13.0+cu132, nelux 0.16.0.
> Software encoders on both sides are libx264 at CRF 20 / preset `medium`; NVENC
> is `p4` at CQ 20. Output sizes are listed because NVENC at CQ 20 is not
> size-matched to x264 at CRF 20 — the **libx264 rows are the apples-to-apples
> software comparison**.

### Summary — end-to-end transcode (Load Video → Save Video)

| Clip | ComfyUI (PyAV) | Nelux (nvdec → h264_nvenc) | Speedup |
|---|---|---|---|
| 720p h264, 300 frames | 6.38 s (47 fps) | **0.45 s (668 fps)** | **14.2x** |
| 1080p h264, 300 frames | 10.17 s (30 fps) | **0.92 s (326 fps)** | **11.1x** |
| 4K h264, 60 frames | 5.43 s (11 fps) | **0.82 s (74 fps)** | **6.7x** |

### 1080p h264 — 1920×1080, 30 fps, 300 frames

| Operation | ComfyUI (PyAV) | Nelux | Speedup |
|---|---|---|---|
| Metadata probe | 5 ms | **4 ms** (`nelux.probe`) | 1.2x |
| Decode → IMAGE | 9.04 s (33 fps) | **1.75 s (172 fps)** — cpu | **5.2x** |
| | | 2.06 s (146 fps) — nvdec | 4.4x |
| Encode ← IMAGE | 6.37 s (47 fps) | **1.61 s (187 fps)** — libx264 | **4.0x** |
| | | 1.07 s (281 fps) — h264_nvenc | 6.0x |
| | | 1.06 s (284 fps) — hevc_nvenc | 6.0x |
| Transcode | 10.17 s (30 fps) | **1.24 s (242 fps)** — nvdec→libx264 | **8.2x** |
| | | **0.92 s (326 fps)** — nvdec→h264_nvenc | **11.1x** |

### 720p h264 — 1280×720, 24 fps, 300 frames

| Operation | ComfyUI (PyAV) | Nelux | Speedup |
|---|---|---|---|
| Metadata probe | 2 ms | **1 ms** | 1.7x |
| Decode → IMAGE | 1.73 s (173 fps) | **0.62 s (481 fps)** — cpu | **2.8x** |
| | | 0.75 s (398 fps) — nvdec | 2.3x |
| Encode ← IMAGE | 4.42 s (68 fps) | **1.14 s (262 fps)** — libx264 | **3.9x** |
| | | 0.47 s (643 fps) — h264_nvenc | 9.5x |
| Transcode | 6.38 s (47 fps) | 0.99 s (304 fps) — nvdec→libx264 | 6.5x |
| | | **0.45 s (668 fps)** — nvdec→h264_nvenc | **14.2x** |

### 4K h264 — 3840×2160, 30 fps, 60 frames

| Operation | ComfyUI (PyAV) | Nelux | Speedup |
|---|---|---|---|
| Metadata probe | 17 ms | 17 ms | 1.0x |
| Decode → IMAGE | 1.74 s (34 fps) | 1.46 s (41 fps) — cpu | 1.2x |
| | | **1.32 s (45 fps)** — nvdec | **1.3x** |
| Encode ← IMAGE | 3.64 s (17 fps) | **1.51 s (40 fps)** — libx264 | **2.4x** |
| | | 0.96 s (63 fps) — h264_nvenc | 3.8x |
| Transcode | 5.43 s (11 fps) | 1.08 s (56 fps) — nvdec→libx264 | 5.0x |
| | | **0.82 s (74 fps)** — nvdec→h264_nvenc | **6.7x** |

The 4K decode gain is small (1.2–1.3x) — at that resolution the wall clock is
dominated by moving 24 MB per frame into a float32 IMAGE tensor, which both
sides pay equally. The encode and transcode gains survive because that is where
NVENC and the streaming pipeline actually do work.

### Peak host memory

Peak RSS for a transcode, each case measured in a **fresh process** (RSS is not
returned to the OS promptly, so a peak sampled after other work mostly reports
that earlier work's high-water mark):

| Clip (frames) | ComfyUI (PyAV) | Nelux cpu→libx264 | Nelux nvdec→h264_nvenc |
|---|---|---|---|
| 720p (200) | 5 697 MB | 1 073 MB (5.3x less) | **782 MB (7.3x less)** |
| 1080p (200) | 11 070 MB | 1 864 MB (5.9x less) | **829 MB (13.4x less)** |
| 4K (40) | 10 015 MB | 2 672 MB (3.8x less) | **1 094 MB (9.2x less)** |

The memory gap is architectural, not a tuning win: Comfy's path materializes
every frame as float32 (`W × H × 12` bytes each — 24 MB per 1080p frame, 95 MB
per 4K frame), while the Nelux transcode streams one frame from demuxer to
encoder at a time. A long clip is bounded by disk rather than by RAM.

### Reproducing

Raw results for every run above are committed under
[`benchmarks/`](benchmarks/).

```bash
python benchmark.py path/to/clip.mp4 --frames 300 --repeats 3 --json bench.json
```

`--frames` sets the timed workload, `--mem-frames` the (smaller) peak-memory
workload. PyAV and psutil are needed for the baseline: `pip install av psutil`.

## Parity with the built-in nodes

The `VIDEO` nodes emit and consume ComfyUI's real `VIDEO` type, so they mix
freely with the stock **Get Video Components**, **Create Video** and **Video
Slice** nodes:

- **Nelux Load Video → Get Video Components** — the component split decodes with
  NVDEC instead of PyAV.
- **Create Video → Nelux Save Video** — the encode runs on NVENC instead of the
  PyAV software H.264 encoder.
- **Nelux Load Video → Nelux Save Video** — full transcode on the GPU. Frames go
  straight from the demuxer to the encoder one at a time, so a long clip is
  bounded by disk rather than by host memory, and an NVDEC surface is handed to
  NVENC without a round trip through system RAM. The source audio track is copied
  through untouched (no re-encode).

Any *other* `VIDEO` implementation goes through Comfy's `get_components()`, which
materializes every frame as float32. Prefer Nelux Load Video as the source when
you are only transcoding.

## Codecs and presets

`codec` accepts `auto` (the default), a generic family name (`h264`, `hevc`,
`av1`), or an exact encoder name (`h264_nvenc`, `libx264`, …).

- `auto` and the generic names prefer NVENC and fall back to the matching
  software encoder when no NVENC device is present.
- An **exact** NVENC name with no NVENC available raises rather than silently
  downgrading.

`preset` is expressed on NVENC's `p1` (fastest) … `p7` (best quality) ladder and
is translated into whatever family the resolved encoder actually speaks — `p4`
becomes `medium` on libx264/libx265 and on QSV. `auto` lets the encoder pick its
own default. FFmpeg's own names (`veryfast`, `slow`, …) and codec-specific values
(`lossless`) are accepted and translated the same way. QSV has no
`ultrafast`/`superfast` rung, so its ladder sits one step slower at the fast end.

> Passing an NVENC preset name straight to a software encoder is an error
> (`x264 [error]: invalid preset 'p4'`), which is why the node normalizes it.

`cq` is the quality knob for every engine, but it does not reach QSV by the same
route. nelux maps `cq` onto `crf`/`qp` for x264/x265, SVT-AV1, libaom and NVENC;
QSV matches none of those branches, so passing `cq` there is **accepted and
silently dropped** — encoding the same clip with and without it gives
byte-identical output. The node therefore forwards QSV's own `global_quality`
AVOption instead, so `cq` behaves consistently across all three engines.

## Audio

- **File-backed video** (Nelux Load Video, or any file `VIDEO`): the original
  audio stream is copied into the output losslessly via nelux `add_passthrough` —
  e.g. `aac` stays `aac`. Works in every container.
- **Component-backed video** (Create Video with an `AUDIO` input): the audio is a
  raw tensor, so it can only be muxed as PCM. Choose the **mkv** or **mov**
  container. Selecting **mp4** raises: FFmpeg's mp4 muxer accepts a `pcm_s16le`
  stream, writes a zero-length `stsd` entry, and reports success — the resulting
  file cannot be demuxed at all (`invalid size 0 in stsd`).

## Known limitations

- **No embedded workflow metadata.** nelux's encoder has no container-metadata
  API, so a video saved through these nodes cannot be dragged back into ComfyUI
  to restore the graph. Use Comfy's built-in Save Video when you need that.
- **8-bit only.** Frames are decoded with `force_8bit` and exposed as float32
  `IMAGE` in `[0, 1]`; 10-bit sources are tone-mapped down on load, and a
  `bit_depth` request other than 8 is ignored with a warning.
- **PyAV is still used for the AUDIO tensor** returned by Get Video Components.
  Video decode never touches it.

## Upstream bugs worked around

Two nelux 0.16.0 defects are handled inside the nodes; both are filed upstream
and the workarounds can be dropped once they land.

- [Nelux#57](https://github.com/NevermindNilas/Nelux/issues/57) — on NVDEC,
  `set_range(start, end)` + iteration returns frames at absolute index
  `keyframe + start`, silently. Any trimmed NVDEC read would return footage from
  the wrong part of the clip. The nodes decode from frame 0 and drop the prefix
  instead of seeking, which stays streaming and stays on the GPU.
- [Nelux#58](https://github.com/NevermindNilas/Nelux/issues/58) — on the CPU
  backend, `decode_batch` / `get_batch_range` converts YUV→RGB with BT.601
  regardless of the stream's declared colour space (up to 40/255 error on a
  bt709 clip). The nodes never use the batch API, costing ~1.9x on strided reads
  that start late in a file but keeping output byte-exact against FFmpeg.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

The tests exercise the node module's Comfy-, PyAV- and nelux-independent surface,
so they run headless in any environment — only torch is required.

## License

AGPL-3.0, matching Nelux.
