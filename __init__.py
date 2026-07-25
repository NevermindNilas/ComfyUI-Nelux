"""Nelux video I/O nodes for ComfyUI.

Drop this folder into ``ComfyUI/custom_nodes``. Nodes appear under the "Nelux"
category and are drop-in faster replacements for Comfy's built-in video nodes:

  - Nelux Load Video    -> VIDEO    (hardware decode; parity with core LoadVideo)
  - Nelux Save Video    -> VIDEO    (NVENC encode + lossless audio passthrough)
  - Nelux Video Info    -> metadata (header only, no decode)
  - Nelux Load Frames   -> IMAGE    (direct frame-range decode to an IMAGE batch)
  - Nelux Encode Frames -> path     (direct IMAGE batch -> file)

The VIDEO nodes emit / consume Comfy's real ``VIDEO`` type, so they interoperate
with the built-in ``Get Video Components`` / ``Create Video`` / ``Video Slice``
nodes -- routing an IMAGE workflow through Nelux Load Video makes the subsequent
decode use NVDEC, and feeding a Create Video result to Nelux Save Video makes the
encode use NVENC.

Decode and encode each pick their own engine, and both default to "auto":

  - ``decode_accelerator``: auto / nvdec / qsv / cpu  (Load Video, Load Frames)
  - ``encoder_engine``:     auto / nvenc / qsv / cpu  (Save Video, Encode Frames)

"auto" probes for working hardware and falls back to the CPU, so the pack works
unchanged on a machine with no GPU; an explicit choice raises rather than
silently downgrading. Availability always comes from opening the real
decoder/encoder -- FFmpeg's build-time codec list says nothing about whether
this machine can run it.

Nelux must be importable in Comfy's Python (torch ABI must match the wheel).

FFmpeg needs no setup: nelux delay-loads it and does not bundle it, so on
Windows these nodes check the DLLs are resolvable and fetch a build once, into a
per-user cache, if they are not. Point NELUX_FFMPEG_DLL_DIR at your own FFmpeg
to use that instead, or set NELUX_NO_AUTO_DOWNLOAD=1 to disable the fetch.
"""

from __future__ import annotations

import ctypes
import logging
import os
import tempfile
import threading
import wave

import torch


_LOG = logging.getLogger(__name__)
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        _LOG.warning("Nelux: %s", message)


# --------------------------------------------------------------------------- #
# Runtime dependency loading (kept lazy so this module imports without Comfy).
#
# The published nelux wheel deliberately does not bundle FFmpeg: its build sets
# NELUX_BUNDLE_FFMPEG_DLLS=OFF so it can share the copy TheAnimeScripter already
# ships. That leaves a bare ComfyUI install with no FFmpeg at all, so the nodes
# fetch one on first use.
#
# The archive is BtbN's win64-gpl-shared build -- the same URL nelux's own wheel
# workflow builds against, which is what keeps the avcodec/avutil sonames and the
# nvenc/qsv feature set matched to the extension. It is fetched unpinned ("latest"
# by request), so the trust boundary is GitHub plus that release: there is no
# hash to verify a downloaded DLL against. Set NELUX_FFMPEG_DLL_DIR to an FFmpeg
# you manage yourself, or NELUX_NO_AUTO_DOWNLOAD=1 to turn the fetch off, if that
# is not acceptable for your install.
# --------------------------------------------------------------------------- #
_DLL_HANDLES = []
_NELUX_MODULE = None
_FFMPEG_LOCK = threading.Lock()

_FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl-shared.zip"
)


def _is_windows() -> bool:
    """Seam so the loader's platform behaviour can be exercised in tests."""
    return os.name == "nt"


def _add_dll_dir(path: str | None) -> None:
    if _is_windows() and path and os.path.isdir(path):
        _DLL_HANDLES.append(os.add_dll_directory(path))


def _configured_dll_dirs() -> list[str]:
    return [
        path
        for path in (os.environ.get("NELUX_FFMPEG_DLL_DIR"),
                     os.environ.get("FFMPEG_DLL_DIR"))
        if path
    ]


def _ffmpeg_cache_dir() -> str:
    """Where a downloaded FFmpeg lives. Kept out of the node directory so a
    read-only or git-managed checkout is never written to."""
    override = os.environ.get("NELUX_FFMPEG_CACHE_DIR")
    if override:
        return override
    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_CACHE_HOME")
        or os.path.expanduser("~/.cache")
    )
    return os.path.join(base, "ComfyUI-Nelux", "ffmpeg")


def _auto_download_enabled() -> bool:
    if not _is_windows():
        return False  # Linux/macOS get FFmpeg from the system package manager
    return os.environ.get("NELUX_NO_AUTO_DOWNLOAD", "").strip().lower() not in (
        "1", "true", "yes", "on",
    )


# nelux DELAY-loads FFmpeg (checked against _nelux.pyd's delay-import
# directory: avcodec-63, avformat-63, avutil-61, swscale-10, swresample-7).
# Delay-loading means `import nelux` SUCCEEDS with no FFmpeg installed at all --
# the loader only looks when a symbol is first called, and then fails as an
# uncatchable process abort (0xC06D007E, the delay-load helper) that takes
# ComfyUI down with it. So there is no ImportError to react to: the DLLs have to
# be checked directly, before anything calls into the extension.
#
# Each entry lists the sonames that satisfy one dependency, newest first --
# mirroring nelux's own fallback table, so a wheel built against either FFmpeg
# generation is accepted.
_REQUIRED_FFMPEG_DLLS = (
    ("avcodec-63.dll", "avcodec-62.dll"),
    ("avformat-63.dll", "avformat-62.dll"),
    ("avutil-61.dll", "avutil-60.dll"),
    ("swscale-10.dll", "swscale-9.dll"),
    ("swresample-7.dll", "swresample-6.dll"),
)


def _can_load_dll(name: str) -> bool:
    """Can the loader resolve `name` right now? ctypes searches the same
    directories os.add_dll_directory() feeds the delay-load helper, and raises
    OSError instead of aborting, which makes this safe to ask."""
    try:
        ctypes.WinDLL(name)
        return True
    except OSError:
        return False


def _missing_ffmpeg_dlls() -> list[str]:
    """The FFmpeg dependencies that cannot currently be resolved, one name per
    unsatisfied dependency."""
    if not _is_windows():
        return []
    return [
        alternatives[0]
        for alternatives in _REQUIRED_FFMPEG_DLLS
        if not any(_can_load_dll(name) for name in alternatives)
    ]


def _download_ffmpeg(destination: str) -> str:
    """Fetch and unpack FFmpeg's runtime DLLs into `destination`/bin.

    Extraction is atomic: everything lands in a sibling temp directory and is
    renamed into place only once complete, so an interrupted download can never
    leave a half-populated directory that later looks like a valid cache."""
    import shutil
    import urllib.request
    import zipfile

    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="ffmpeg-dl-", dir=parent)
    archive = os.path.join(staging, "ffmpeg.zip")

    _LOG.info("Nelux: downloading FFmpeg runtime DLLs from %s", _FFMPEG_URL)
    try:
        request = urllib.request.Request(
            _FFMPEG_URL, headers={"User-Agent": "ComfyUI-Nelux"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(archive, "wb") as out:
            shutil.copyfileobj(response, out)

        bin_dir = os.path.join(staging, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            members = [
                name for name in zf.namelist()
                if name.lower().endswith(".dll") and "/bin/" in name.replace("\\", "/")
            ]
            if not members:
                raise RuntimeError(
                    "the FFmpeg archive contained no bin/*.dll entries"
                )
            for name in members:
                # Flatten: take the basename only, so a crafted path inside the
                # archive cannot escape the destination directory.
                target = os.path.join(bin_dir, os.path.basename(name))
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        if not any(f.startswith("avcodec-") for f in os.listdir(bin_dir)):
            raise RuntimeError("the FFmpeg archive contained no avcodec DLL")

        os.remove(archive)
        if os.path.isdir(destination):
            shutil.rmtree(destination, ignore_errors=True)
        os.replace(staging, destination)
        _LOG.info("Nelux: FFmpeg installed to %s", destination)
        return os.path.join(destination, "bin")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _ensure_ffmpeg() -> None:
    """Make nelux's delay-loaded FFmpeg resolvable, downloading it if needed.

    Order: whatever the user configured, then whatever the system already
    provides, then a previously downloaded copy, then a fresh download."""
    for path in _configured_dll_dirs():
        _add_dll_dir(path)
    if not _missing_ffmpeg_dlls():
        return

    cached = os.path.join(_ffmpeg_cache_dir(), "bin")
    if os.path.isdir(cached):
        _add_dll_dir(cached)
        if not _missing_ffmpeg_dlls():
            return

    missing = _missing_ffmpeg_dlls()
    if not _auto_download_enabled():
        raise ImportError(
            f"Nelux needs FFmpeg, and these DLLs cannot be found: "
            f"{', '.join(missing)}.\nAutomatic download is disabled "
            f"(NELUX_NO_AUTO_DOWNLOAD), so set NELUX_FFMPEG_DLL_DIR to a "
            f"directory containing them."
        )

    _LOG.info("Nelux: FFmpeg not found (%s); fetching it once",
              ", ".join(missing))
    try:
        _add_dll_dir(_download_ffmpeg(_ffmpeg_cache_dir()))
    except Exception as exc:  # network down, proxy, disk full, ...
        raise ImportError(
            f"Nelux needs FFmpeg ({', '.join(missing)}) and downloading it "
            f"failed: {exc}\nSet NELUX_FFMPEG_DLL_DIR to a directory holding "
            f"the FFmpeg DLLs, or fix network access and retry."
        ) from exc

    still_missing = _missing_ffmpeg_dlls()
    if still_missing:
        raise ImportError(
            f"Nelux still cannot load {', '.join(still_missing)} after "
            f"downloading FFmpeg into {_ffmpeg_cache_dir()}. The build fetched "
            f"probably does not match the sonames this Nelux wheel was linked "
            f"against; set NELUX_FFMPEG_DLL_DIR to a matching FFmpeg."
        )


def _import_nelux():
    """Import nelux with its FFmpeg dependency guaranteed present.

    The FFmpeg check has to happen here rather than around the import: nelux
    delay-loads FFmpeg, so the import succeeds either way and the failure only
    lands later, as a process abort inside the first decode."""
    global _NELUX_MODULE
    if _NELUX_MODULE is not None:
        return _NELUX_MODULE

    with _FFMPEG_LOCK:
        if _NELUX_MODULE is not None:
            return _NELUX_MODULE

        _ensure_ffmpeg()
        try:
            import nelux
        except ImportError as exc:
            raise ImportError(
                "Nelux failed to import in Comfy's Python. The wheel most "
                "likely does not match this Python or this PyTorch minor "
                "version -- Comfy Desktop's bundled torch often differs from "
                "the published wheel's ABI. Install a Nelux wheel built for "
                "this combination."
            ) from exc

        _NELUX_MODULE = nelux
        return nelux


def _comfy_video_api():
    """Return (Input, InputImpl, Types) from Comfy's public video API."""
    try:
        from comfy_api.latest import Input, InputImpl, Types  # type: ignore

        return Input, InputImpl, Types
    except ImportError as exc:  # pragma: no cover - requires ComfyUI runtime
        raise ImportError(
            "Comfy's video API (comfy_api.latest) is unavailable. These nodes "
            "require a ComfyUI build new enough to expose the VIDEO type."
        ) from exc


def _folder_paths():
    import folder_paths  # type: ignore

    return folder_paths


# --------------------------------------------------------------------------- #
# Metadata
#
# nelux.probe() reads the container header without initializing a decoder,
# allocating frame buffers, or starting worker threads -- much cheaper than
# opening a VideoReader just to read width/height/fps, and it avoids the process
# spawn an external ffprobe call pays.
# --------------------------------------------------------------------------- #
def _probe(path: str) -> dict:
    return _import_nelux().probe(path)


# --------------------------------------------------------------------------- #
# Decode accelerator resolution
#
# nelux has no "auto", so the node resolves it: a plain default of "nvdec" would
# hard-fail on any machine without an NVIDIA GPU.
#
# Which backends exist depends on the nelux build. Every release has 'cpu' and
# 'nvdec'; 'qsv' decode is NOT in nelux 0.16.0 (its Factory.hpp maps only CPU and
# NVDEC) and is rejected at construction there. It stays on the list because the
# option is cheap and forward-looking -- "auto" simply never selects a backend it
# could not open, and an explicit pick gets a message that says what happened.
# QSV *encode* is unrelated and does work: that lives in FFmpeg, not in a nelux
# decode backend.
# --------------------------------------------------------------------------- #
_ACCELERATORS = ["auto", "nvdec", "qsv", "cpu"]
_HARDWARE_ACCELERATORS = ("nvdec", "qsv")
_ACCEL_CACHE: dict[str, str] = {}


def _accelerator_works(nelux, path: str, accelerator: str) -> bool:
    """Can this machine decode `path` with `accelerator`? Probed exactly the way
    the nodes decode, since force_8bit changes which pixel formats a backend has
    to handle and so can change the answer."""
    try:
        with nelux.VideoReader(path, force_8bit=True,
                               decode_accelerator=accelerator):
            return True
    except Exception:
        return False


def _resolve_accelerator(path: str, choice: str) -> str:
    """Map the node's accelerator choice onto one nelux accepts. "auto" probes
    for a working hardware decoder once per codec and remembers the answer;
    an explicit choice is passed through, so it raises rather than silently
    downgrading -- but with a message that names the actual problem."""
    name = (choice or "auto").strip().lower()
    nelux = _import_nelux()

    if name != "auto":
        if name in _HARDWARE_ACCELERATORS and not _accelerator_works(
            nelux, path, name
        ):
            raise RuntimeError(
                f"Nelux: decode_accelerator='{name}' cannot decode this file on "
                f"this machine. Either the hardware/driver is missing, or this "
                f"nelux build has no {name} decode backend (0.16.0 ships 'cpu' "
                f"and 'nvdec' only). Use 'auto' to take whatever is present."
            )
        return name

    try:
        key = str(_probe(path).get("codec", "?"))
    except Exception:
        key = "?"
    cached = _ACCEL_CACHE.get(key)
    if cached is not None:
        return cached
    for candidate in _HARDWARE_ACCELERATORS:
        if _accelerator_works(nelux, path, candidate):
            _ACCEL_CACHE[key] = candidate
            return candidate
    _ACCEL_CACHE[key] = "cpu"
    return "cpu"


# --------------------------------------------------------------------------- #
# Trimmed iteration
#
# nelux bug (0.16.0), NevermindNilas/Nelux#57: on the nvdec backend,
# set_range(start, end) followed by
# iteration seeks to the enclosing keyframe K and then *also* discards `start`
# frames, so it yields absolute frames K + start .. instead of start .. -- and
# it does so silently. Verified with a marker clip whose frame index is encoded
# in its pixels:
#
#   GOP 250, set_range(300, 303), nvdec -> frames 550, 551, 552
#   GOP 250, set_range(500, 503), nvdec -> nothing at all (500 + 500 is past EOF)
#
# start < the first keyframe interval is unaffected (K == 0), as is the CPU
# backend, and the batch APIs (get_batch/get_batch_range) are correct on both.
# So on nvdec we decode from 0 and drop the prefix: still streaming, still on
# the GPU, and correct. The prefix decode is cheap relative to NVDEC throughput.
# --------------------------------------------------------------------------- #
def _iter_range(reader, start: int, end: int, accelerator: str):
    """Yield frames [start, end) from `reader`, working around the nvdec seek
    offset bug."""
    if accelerator == "nvdec" and start > 0:
        reader.set_range(0, end)
        for index, frame in enumerate(reader):
            if index >= start:
                yield frame
        return
    reader.set_range(start, end)
    yield from reader


# --------------------------------------------------------------------------- #
# Codec / preset resolution
#
# nelux forwards a *string* preset straight to av_dict_set("preset", ...), so a
# preset name from the wrong family (e.g. NVENC's "p4" on libx264) makes
# avcodec_open2 fail with "Invalid argument". Everything below normalizes a
# single user-facing preset onto whatever family the resolved encoder speaks.
# --------------------------------------------------------------------------- #
_NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc", "av1_nvenc"})
_QSV_CODECS = frozenset({"h264_qsv", "hevc_qsv", "av1_qsv", "vp9_qsv", "mpeg2_qsv"})
_X26X_CODECS = frozenset({"libx264", "libx264rgb", "libx265"})

# Every accepted preset name collapses onto a 1 (fastest) .. 7 (best) rung.
_PRESET_RUNGS = {
    "p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5, "p6": 6, "p7": 7,
    "ultrafast": 1, "superfast": 1, "veryfast": 2, "faster": 3, "fast": 3,
    "medium": 4, "slow": 5, "slower": 6, "veryslow": 7,
    "fastest": 1, "slowest": 7,
}
_X26X_BY_RUNG = {
    1: "ultrafast", 2: "veryfast", 3: "faster", 4: "medium",
    5: "slow", 6: "slower", 7: "veryslow",
}
# QSV has no "ultrafast"/"superfast" rung, so its seven names sit one step
# slower than the x26x ladder at the fast end.
_QSV_BY_RUNG = {
    1: "veryfast", 2: "faster", 3: "fast", 4: "medium",
    5: "slow", 6: "slower", 7: "veryslow",
}

# The three engines a user can pick, fastest-hardware-first. "cpu" is last and
# is the only one guaranteed to exist.
_ENGINES = ["auto", "nvenc", "qsv", "cpu"]
_ENGINE_ORDER = ("nvenc", "qsv", "cpu")

# Codec family -> the encoder each engine uses for it.
_ENCODER_TABLE = {
    "h264": {"nvenc": "h264_nvenc", "qsv": "h264_qsv", "cpu": "libx264"},
    "hevc": {"nvenc": "hevc_nvenc", "qsv": "hevc_qsv", "cpu": "libx265"},
    "av1": {"nvenc": "av1_nvenc", "qsv": "av1_qsv", "cpu": "libsvtav1"},
}
# Everything a user (or Comfy's VideoCodec enum) might name a family.
_FAMILY_ALIASES = {
    "auto": "h264", "h264": "h264", "avc1": "h264", "avc": "h264",
    "h265": "hevc", "hevc": "hevc", "hvc1": "hevc",
    "av1": "av1", "av01": "av1",
}


def _resolve_preset(encoder: str, preset) -> str | None:
    """Translate a preset name into the family `encoder` accepts, or None to let
    the encoder pick its own default. Unrecognized names pass through verbatim so
    codec-specific values (e.g. libsvtav1's "8", NVENC's "lossless") still work."""
    if preset is None:
        return None
    name = str(preset).strip().lower()
    if not name or name == "auto":
        return None
    rung = _PRESET_RUNGS.get(name)
    if rung is None:
        return str(preset).strip()
    if encoder in _NVENC_CODECS:
        return f"p{rung}"
    if encoder in _QSV_CODECS:
        return _QSV_BY_RUNG[rung]
    if encoder in _X26X_CODECS:
        return _X26X_BY_RUNG[rung]
    return None


# --------------------------------------------------------------------------- #
# Encoder availability
#
# nelux.get_available_encoders()/get_nvenc_encoders() enumerate what FFmpeg was
# *compiled* with (av_codec_iterate + a name filter) -- they say nothing about
# whether this machine can actually run the encoder. This build lists NVENC,
# QSV, AMF and VAAPI encoders on every platform, so trusting that list makes
# "auto" pick h264_nvenc on an AMD-only box and then die inside avcodec_open2.
#
# The only honest check is to open the encoder. It also catches per-GPU limits a
# name list never could: av1_nvenc is present in the build and fails on an
# RTX 3090, because AV1 encode starts at Ada.
#
# 256x256 clears NVENC's minimum frame dimensions (it rejects 128x128). Probes
# cost ~80 ms for NVENC and ~500 ms for QSV, so each answer is cached and only
# hardware encoders are ever probed.
# --------------------------------------------------------------------------- #
_ENCODER_PROBE_CACHE: dict[str, bool] = {}
_PROBE_SIZE = (256, 256)


def _is_hardware_encoder(encoder: str) -> bool:
    return encoder in _NVENC_CODECS or encoder in _QSV_CODECS


def _encoder_available(nelux, encoder: str) -> bool:
    """True if this machine can actually open `encoder`. Software encoders are
    taken on trust; hardware encoders are probed once and remembered."""
    if not _is_hardware_encoder(encoder):
        return True
    cached = _ENCODER_PROBE_CACHE.get(encoder)
    if cached is not None:
        return cached
    width, height = _PROBE_SIZE
    handle, probe_path = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    try:
        with nelux.VideoEncoder(probe_path, codec=encoder, width=width,
                                height=height, fps=30.0):
            pass
        available = True
    except Exception:
        available = False
    finally:
        try:
            os.remove(probe_path)
        except OSError:
            pass
    _ENCODER_PROBE_CACHE[encoder] = available
    return available


def _resolve_encoder(nelux, codec: str, engine: str = "auto") -> str:
    """Map a (codec, engine) choice onto a concrete encoder name.

    `codec` is either a family ("auto", "h264", "hevc", "av1") or an exact
    encoder name ("h264_nvenc", "libx265", ...); an exact name wins outright and
    makes `engine` irrelevant. For a family, an explicit `engine` is honoured as
    written -- and raises if that engine cannot encode it here, rather than
    silently downgrading -- while "auto" walks NVENC, then QSV, then the CPU."""
    name = (codec or "auto").strip().lower()
    engine_name = (engine or "auto").strip().lower()
    family = _FAMILY_ALIASES.get(name)

    if family is None:  # an exact encoder name
        if engine_name != "auto":
            _warn_once(
                f"engine-ignored-{name}",
                f"codec '{name}' names an exact encoder, so encoder_engine="
                f"'{engine_name}' is ignored. Pick a codec family (auto/h264/"
                f"hevc/av1) to choose the engine.",
            )
        if _is_hardware_encoder(name) and not _encoder_available(nelux, name):
            raise RuntimeError(
                f"Nelux: {name} was requested but this machine cannot open it "
                f"(no such device, or the GPU does not support that codec -- "
                f"av1_nvenc needs Ada or newer, for example). Use 'auto' to fall "
                f"back to whatever is present."
            )
        return name

    by_engine = _ENCODER_TABLE[family]
    if engine_name != "auto":
        if engine_name not in by_engine:
            raise RuntimeError(
                f"Nelux: unknown encoder_engine '{engine_name}'. "
                f"Expected one of {_ENGINES}."
            )
        encoder = by_engine[engine_name]
        if not _encoder_available(nelux, encoder):
            raise RuntimeError(
                f"Nelux: encoder_engine='{engine_name}' cannot encode {family} "
                f"on this machine ({encoder} failed to open). Use 'auto', or "
                f"pick a different engine."
            )
        return encoder

    for candidate_engine in _ENGINE_ORDER:
        encoder = by_engine[candidate_engine]
        if _encoder_available(nelux, encoder):
            return encoder
    return by_engine["cpu"]  # unreachable: software is never probed


def _encoder_kwargs(nelux, codec: str, width: int, height: int, fps: float,
                    preset, cq: int, engine: str = "auto") -> dict:
    encoder = _resolve_encoder(nelux, codec, engine)
    kwargs = {
        "codec": encoder,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }
    if encoder in _QSV_CODECS:
        # nelux maps `cq` onto crf/qp for x26x, SVT-AV1, libaom and NVENC, but
        # QSV matches none of those branches, so `cq` is accepted and silently
        # dropped -- verified by encoding the same clip with and without it and
        # getting byte-identical output. QSV's equivalent knob is the
        # global_quality AVOption, which `options` forwards to avcodec_open2.
        kwargs["options"] = {"global_quality": str(int(cq))}
    else:
        kwargs["cq"] = int(cq)
    resolved_preset = _resolve_preset(encoder, preset)
    if resolved_preset is not None:
        kwargs["preset"] = resolved_preset
    return kwargs


# --------------------------------------------------------------------------- #
# Comfy-independent cores (unit-testable without ComfyUI or PyAV).
# --------------------------------------------------------------------------- #
def _to_comfy_image(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Nelux frames must be [H,W,3]")
    out = frame.to(torch.float32)
    if out is frame:  # already float32: don't scale the decoder's own buffer
        out = out.clone()
    return out.div_(255)


def _to_nelux_frame(image: torch.Tensor) -> torch.Tensor:
    # Device is deliberately preserved: a CUDA IMAGE fed to an NVENC codec stays
    # on the GPU through the whole encode (VideoEncoder has a CUDA fast path).
    # The encoder downloads to host itself for software codecs.
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Comfy IMAGE frames must be [H,W,3]")
    return image.clamp(0, 1).mul(255).to(torch.uint8).contiguous()


def _validate_path(input_path: str):
    if not input_path:
        return "Nelux: input_path is empty"
    if not os.path.isfile(input_path):
        return f"Nelux: file not found: {input_path}"
    return True


def _file_signature(input_path: str) -> float:
    try:
        return os.path.getmtime(input_path)
    except OSError:
        return float("nan")


def _frame_range(fps: float, total_frames: int, start_time: float, duration: float):
    """Convert a (start_time, duration) window in seconds to a half-open frame
    range. duration == 0 means "to the end of the clip"."""
    start_f = max(0, int(round(start_time * fps)))
    if duration:
        end_f = start_f + max(1, int(round(duration * fps)))
    elif total_frames > 0:
        end_f = int(total_frames)
    else:
        end_f = 2**31 - 1  # header has no frame count: read to EOF
    if total_frames > 0:
        end_f = min(end_f, int(total_frames))
        start_f = min(start_f, max(0, int(total_frames) - 1))
    return start_f, max(end_f, start_f + 1)


def _decode_frames_nelux(
    path: str,
    decode_accelerator: str = "cpu",
    start_time: float = 0.0,
    duration: float = 0.0,
    resize=None,
):
    """Decode a (possibly trimmed) clip to an IMAGE batch. Returns
    (images[B,H,W,3] float32 CPU in [0,1], fps float, width, height).

    This materializes the whole clip as float32, which is what a Comfy IMAGE
    batch requires. Transcodes should use _transcode_nelux instead."""
    nelux = _import_nelux()
    accelerator = _resolve_accelerator(path, decode_accelerator)
    with nelux.VideoReader(
        path,
        force_8bit=True,
        decode_accelerator=accelerator,
        resize=resize,
    ) as reader:
        fps = float(reader.fps)
        width, height = int(reader.width), int(reader.height)
        if start_time or duration:
            start_f, end_f = _frame_range(
                fps, int(reader.total_frames), start_time, duration
            )
            source = _iter_range(reader, start_f, end_f, accelerator)
        else:
            source = reader
        frames = [_to_comfy_image(f) for f in source]
    if not frames:
        raise RuntimeError(f"Nelux decoded no frames from {path}")
    images = torch.stack(frames).cpu()
    return images, fps, width, height


def _transcode_nelux(
    src: str,
    out_path: str,
    codec: str = "auto",
    preset="auto",
    cq: int = 20,
    encoder_engine: str = "auto",
    decode_accelerator: str = "cpu",
    start_time: float = 0.0,
    duration: float = 0.0,
    audio: bool = True,
    subtitles: bool = False,
):
    """Stream src -> out_path one frame at a time, copying audio/subtitles
    without re-encoding.

    Unlike a decode-to-IMAGE-batch-then-encode round trip this never holds more
    than one frame, so a long clip cannot exhaust host memory, and an NVDEC frame
    is handed straight to NVENC without ever leaving the GPU."""
    nelux = _import_nelux()
    accelerator = _resolve_accelerator(src, decode_accelerator)
    with nelux.VideoReader(
        src, force_8bit=True, decode_accelerator=accelerator
    ) as reader:
        fps = float(reader.fps)
        trim = None
        source = reader
        if start_time or duration:
            start_f, end_f = _frame_range(fps, int(reader.total_frames), start_time, duration)
            source = _iter_range(reader, start_f, end_f, accelerator)
            trim = (start_f / fps, end_f / fps)
        copy_audio = bool(audio) and bool(reader.has_audio)

        kwargs = _encoder_kwargs(
            nelux, codec, int(reader.width), int(reader.height), fps, preset, cq,
            engine=encoder_engine,
        )
        with nelux.VideoEncoder(out_path, **kwargs) as enc:
            if copy_audio or subtitles:
                passthrough = {"audio": copy_audio, "subtitles": bool(subtitles)}
                if trim is not None:
                    passthrough["start"], passthrough["end"] = trim
                enc.add_passthrough(src, **passthrough)
            for frame in source:
                enc.encode_frame(frame)
    return out_path


def _write_waveform_wav(audio: dict, wav_path: str) -> bool:
    """Write a Comfy AUDIO dict ({"waveform": [B,C,T], "sample_rate": int}) to a
    PCM16 WAV so it can be muxed into an encode via add_passthrough. Returns True
    if audio was written, False if there is nothing to write.

    Note: add_passthrough stream-copies this PCM, so a component-audio VIDEO
    (e.g. from Create Video) saved to mp4/mov ends up with pcm_s16le audio --
    playable in ffmpeg/VLC but not most browsers. The common path (a file-backed
    Nelux Load Video -> Nelux Save Video) copies the source's already-compressed
    audio (e.g. aac) untouched and has no such caveat."""
    if not audio:
        return False
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0 or waveform.numel() == 0:
        return False
    wf = waveform
    if wf.ndim == 3:  # [B, C, T] -> take first item
        wf = wf[0]
    if wf.ndim == 1:  # [T] -> [1, T]
        wf = wf.unsqueeze(0)
    channels = int(wf.shape[0])
    # [C, T] float [-1,1] -> [T, C] int16 interleaved.
    pcm = (
        wf.transpose(0, 1).clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).cpu().contiguous()
    )
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.numpy().tobytes())
    return True


# FFmpeg's mp4 muxer accepts a pcm_s16le stream and then writes a zero-length
# stsd sample entry, producing a file no demuxer can open -- and it reports no
# error while doing it. Only offer containers that really hold raw PCM.
_PCM_SAFE_CONTAINERS = frozenset({".mkv", ".mov", ".mka", ".wav"})


def _pcm_container_error(out_path: str) -> str | None:
    """Return an error message if out_path's container cannot hold raw PCM."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext in _PCM_SAFE_CONTAINERS:
        return None
    return (
        f"Nelux: raw PCM audio cannot be muxed into a '{ext or '<no extension>'}' "
        f"container -- FFmpeg silently writes a corrupt file. Choose the mkv or mov "
        f"container, or drive this from a file-backed VIDEO (Nelux Load Video), "
        f"whose audio is already compressed and copies into mp4 cleanly."
    )


def _check_pcm_container(out_path: str, passthrough_source: str | None) -> None:
    if not passthrough_source or not passthrough_source.lower().endswith(".wav"):
        return
    error = _pcm_container_error(out_path)
    if error:
        raise ValueError(error)


def _nelux_encode(
    images: torch.Tensor,
    fps: float,
    out_path: str,
    codec: str = "auto",
    preset="auto",
    cq: int = 20,
    encoder_engine: str = "auto",
    passthrough_source: str | None = None,
    passthrough_subtitles: bool = False,
):
    """Encode an IMAGE batch [B,H,W,3] to out_path via nelux. Optionally copy
    audio/subtitles from passthrough_source (no re-encode)."""
    # Validate before importing nelux: a bad container choice should surface as
    # the actionable "pick mkv or mov" error even on an install where nelux
    # itself cannot load.
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("IMAGE batch must be [B,H,W,3]")
    _check_pcm_container(out_path, passthrough_source)
    nelux = _import_nelux()
    kwargs = _encoder_kwargs(
        nelux, codec, int(images.shape[2]), int(images.shape[1]), fps, preset, cq,
        engine=encoder_engine,
    )
    with nelux.VideoEncoder(out_path, **kwargs) as enc:
        if passthrough_source:
            enc.add_passthrough(
                passthrough_source, audio=True, subtitles=passthrough_subtitles
            )
        for image in images:
            enc.encode_frame(_to_nelux_frame(image))
    return out_path


def _decode_audio_av(path: str, start_time: float = 0.0, duration: float = 0.0):
    """Decode an audio track to a Comfy AUDIO dict via PyAV. Returns None when
    there is no audio track or PyAV is unavailable."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError:
        return None
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            sample_rate = int(stream.rate)
            resampler = AudioResampler(format="fltp", layout=stream.layout)
            chunks = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(torch.from_numpy(out.to_ndarray()))
            for out in resampler.resample(None):  # flush the resampler's tail
                chunks.append(torch.from_numpy(out.to_ndarray()))
            if not chunks:
                return None
            waveform = torch.cat(chunks, dim=1)  # [channels, samples]
            if start_time or duration:
                s = max(0, int(round(start_time * sample_rate)))
                e = s + int(round(duration * sample_rate)) if duration else waveform.shape[1]
                waveform = waveform[:, s:e]
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception:
        return None


def _comfy_codec_name(codec) -> str:
    """Normalize Comfy's VideoCodec enum (or a plain string) to a codec key."""
    if codec is None:
        return "auto"
    name = getattr(codec, "value", codec)
    return str(name).strip().lower() or "auto"


# --------------------------------------------------------------------------- #
# Nelux-backed VIDEO type (built lazily so Comfy is only required at run time).
# --------------------------------------------------------------------------- #
_NELUX_VIDEO_CLS = None


def _nelux_video_class():
    global _NELUX_VIDEO_CLS
    if _NELUX_VIDEO_CLS is not None:
        return _NELUX_VIDEO_CLS

    from fractions import Fraction

    Input, _InputImpl, Types = _comfy_video_api()

    class NeluxVideoFromFile(Input.Video):
        """A file-backed VIDEO whose frames decode through NVDEC. Metadata comes
        from the container header (cheap); audio uses PyAV; save re-encodes with
        NVENC and copies the source audio losslessly."""

        def __init__(self, path: str, decode_accelerator: str = "cpu",
                     start_time: float = 0.0, duration: float = 0.0):
            self._path = path
            self._accel = decode_accelerator
            self._start_time = float(start_time)
            self._duration = float(duration)
            self._props_cache = None

        # -- cheap header-only metadata (no frame decode) -------------------- #
        def _props(self):
            if self._props_cache is None:
                p = _probe(self._path)
                self._props_cache = (
                    int(p["width"]), int(p["height"]), float(p["fps"]),
                    int(p["total_frames"]), float(p["duration"]),
                )
            return self._props_cache

        def get_dimensions(self):
            w, h, _, _, _ = self._props()
            return w, h

        def get_frame_rate(self) -> Fraction:
            _, _, fps, _, _ = self._props()
            return Fraction(fps).limit_denominator(1000000)

        def get_duration(self) -> float:
            _, _, _, _, total_duration = self._props()
            remaining = max(0.0, total_duration - self._start_time)
            return min(self._duration, remaining) if self._duration else remaining

        def get_frame_count(self) -> int:
            _, _, fps, total_frames, _ = self._props()
            if not self._start_time and not self._duration:
                return total_frames
            return max(1, int(round(self.get_duration() * fps)))

        def get_stream_source(self):
            return self._path

        def get_components(self):
            images, fps, _, _ = _decode_frames_nelux(
                self._path, self._accel, self._start_time, self._duration
            )
            audio = _decode_audio_av(self._path, self._start_time, self._duration)
            return Types.VideoComponents(
                images=images,
                frame_rate=Fraction(fps).limit_denominator(1000000),
                audio=audio,
            )

        def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
            """Trim relative to this video's already-trimmed window. The child
            window is clamped to what the parent still has left, so nesting two
            trims never yields more footage than the outer one exposed."""
            offset = float(start_time or 0.0)
            remaining = max(0.0, self.get_duration() - offset)
            requested = float(duration) if duration else 0.0

            if requested > 0:
                if strict_duration and requested > remaining + 1e-6:
                    return None
                child_duration = min(requested, remaining)
            elif self._duration:
                # Parent is bounded, so the child must stay bounded too.
                child_duration = remaining
            else:
                child_duration = 0.0  # both unbounded: run to end of file

            return NeluxVideoFromFile(
                self._path,
                self._accel,
                start_time=self._start_time + offset,
                duration=child_duration,
            )

        def save_to(self, path, format=None, codec=None, metadata=None, bit_depth=None):
            # `format` is honored implicitly: the caller encodes the container
            # choice in `path`'s extension, which is what nelux muxes to.
            if metadata:
                _warn_once(
                    "save-metadata",
                    "nelux's encoder cannot write container metadata, so the saved "
                    "video will not carry the embedded workflow/prompt. Use Comfy's "
                    "built-in Save Video if you need to drag the file back in.",
                )
            if bit_depth not in (None, 8):
                _warn_once(
                    "save-bit-depth",
                    f"bit_depth={bit_depth} requested but Nelux nodes decode 8-bit; "
                    f"saving 8-bit.",
                )
            _transcode_nelux(
                self._path,
                str(path),
                codec=_comfy_codec_name(codec),
                decode_accelerator=self._accel,
                start_time=self._start_time,
                duration=self._duration,
                audio=True,
                subtitles=True,
            )

    _NELUX_VIDEO_CLS = NeluxVideoFromFile
    return _NELUX_VIDEO_CLS


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv")
_PRESETS = ["auto", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
# Families first -- those pair with `encoder_engine` to pick the encoder. The
# exact encoder names below still work and override the engine outright, which
# also keeps workflows saved before `encoder_engine` existed loading unchanged.
_CODECS = [
    "auto", "h264", "hevc", "av1",
    "h264_nvenc", "hevc_nvenc", "av1_nvenc",
    "h264_qsv", "hevc_qsv", "av1_qsv",
    "libx264", "libx265", "libsvtav1",
]


def _input_video_files():
    try:
        folder_paths = _folder_paths()
        d = folder_paths.get_input_directory()
        return sorted(
            f for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(_VIDEO_EXTS)
        )
    except Exception:
        return []


def _resolve_input_path(file: str) -> str:
    try:
        folder_paths = _folder_paths()
        return folder_paths.get_annotated_filepath(file)
    except Exception:
        return file


def _resolve_output_path(output_path: str) -> str:
    """Anchor a relative output path in Comfy's output directory. Without this a
    bare filename lands in whatever directory the Comfy process was started from,
    which is rarely where the user expects to find it. Absolute paths are the
    user's own choice and pass through."""
    path = str(output_path or "").strip()
    if not path:
        raise ValueError("Nelux: output_path is empty")
    if os.path.isabs(path):
        return path
    try:
        base = _folder_paths().get_output_directory()
    except Exception:
        return path
    resolved = os.path.normpath(os.path.join(base, path))
    os.makedirs(os.path.dirname(resolved) or base, exist_ok=True)
    return resolved


class NeluxLoadVideo:
    """Load a video file as a Comfy VIDEO, decoding frames with NVDEC."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_input_video_files(), {"video_upload": True}),
                "decode_accelerator": (_ACCELERATORS,),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "Nelux"

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        return _validate_path(_resolve_input_path(file))

    @classmethod
    def IS_CHANGED(cls, file, **kwargs):
        return _file_signature(_resolve_input_path(file))

    def load(self, file, decode_accelerator):
        path = _resolve_input_path(file)
        video = _nelux_video_class()(path, decode_accelerator=decode_accelerator)
        return (video,)


class NeluxSaveVideo:
    """Encode a Comfy VIDEO to the output directory with NVENC. Audio is copied
    losslessly from the source (file-backed VIDEO) or muxed from the VIDEO's
    audio track (component-backed VIDEO).

    A Nelux VIDEO is transcoded as a stream, so clip length is bounded by disk,
    not by host memory. Any other VIDEO must go through Comfy's get_components(),
    which materializes every frame as float32."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "nelux/video"}),
                "codec": (_CODECS,),
                "encoder_engine": (_ENGINES,),
                "container": (["mp4", "mkv", "mov"],),
                "preset": (_PRESETS,),
                "cq": ("INT", {"default": 20, "min": 0, "max": 51}),
                "audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "save"
    CATEGORY = "Nelux"
    OUTPUT_NODE = True

    def save(self, video, filename_prefix, codec, encoder_engine, container,
             preset, cq, audio):
        folder_paths = _folder_paths()
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        )
        base = f"{filename}_{counter:05}.{container}"
        out_path = os.path.join(full_output_folder, base)

        # A Nelux VIDEO knows its own source and trim window, so it can be piped
        # frame-by-frame from demuxer to encoder. A foreign VIDEO's trim state is
        # private, so its stream source cannot be trusted -- decode it instead.
        if isinstance(video, _nelux_video_class()):
            _transcode_nelux(
                video.get_stream_source(), out_path,
                codec=codec, preset=preset, cq=cq, encoder_engine=encoder_engine,
                decode_accelerator=video._accel,
                start_time=video._start_time, duration=video._duration,
                audio=audio, subtitles=False,
            )
            return self._result(video, base, subfolder)

        components = video.get_components()
        images = components.images
        fps = float(components.frame_rate)

        temp_wav = None
        passthrough_source = None
        if audio and getattr(components, "audio", None):
            # This VIDEO carries an audio *tensor*, not a compressed stream, so
            # it can only be muxed as raw PCM. Fail before encoding anything.
            error = _pcm_container_error(out_path)
            if error:
                raise ValueError(error)
            handle, temp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            if _write_waveform_wav(components.audio, temp_wav):
                passthrough_source = temp_wav

        try:
            _nelux_encode(
                images, fps, out_path,
                codec=codec, preset=preset, cq=cq, encoder_engine=encoder_engine,
                passthrough_source=passthrough_source,
            )
        finally:
            if temp_wav and os.path.isfile(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

        return self._result(video, base, subfolder)

    @staticmethod
    def _result(video, base, subfolder):
        return {
            "ui": {
                "images": [{"filename": base, "subfolder": subfolder, "type": "output"}],
                "animated": (True,),
            },
            "result": (video,),
        }


class NeluxVideoInfo:
    """Probe fps / dimensions / frame count / codec / audio from a file.

    Backed by ``nelux.probe``, which parses the container header only -- no
    decoder init, no frame buffers, no worker threads, and no ffprobe process
    spawn."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = (
        "INT", "INT", "FLOAT", "INT", "FLOAT", "STRING", "BOOLEAN", "STRING", "INT",
    )
    RETURN_NAMES = (
        "width", "height", "fps", "total_frames", "duration", "pixel_format",
        "has_audio", "codec", "bit_depth",
    )
    FUNCTION = "probe"
    CATEGORY = "Nelux"

    @classmethod
    def VALIDATE_INPUTS(cls, input_path):
        return _validate_path(input_path)

    @classmethod
    def IS_CHANGED(cls, input_path="", **kwargs):
        return _file_signature(input_path)

    def probe(self, input_path):
        p = _probe(input_path)
        return (
            int(p["width"]),
            int(p["height"]),
            float(p["fps"]),
            int(p["total_frames"]),
            float(p["duration"]),
            str(p["pixel_format"]),
            bool(p["has_audio"]),
            str(p["codec"]),
            int(p["bit_depth"]),
        )


def _load_frames_plan(total_frames: int, start_frame: int, frame_count: int, step: int):
    """Clamp a (start, count, step) request to what the file actually holds.

    Returns (start, end, count) with `end` exclusive and covering exactly the
    sampled frames. Asking for more frames than remain yields fewer frames rather
    than an out-of-range set_range/get_batch_range."""
    step = max(1, int(step))
    start = max(0, int(start_frame))
    count = max(1, int(frame_count))
    if total_frames > 0:
        start = min(start, int(total_frames) - 1)
        available = int(total_frames) - start
        count = max(1, min(count, -(-available // step)))  # ceil division
    return start, start + (count - 1) * step + 1, count


class NeluxLoadFrames:
    """Direct frame-range decode to an IMAGE batch (no VIDEO wrapper)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 4096}),
                "step": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "decode_accelerator": (_ACCELERATORS,),
                # "cpu" is the safe default: a Comfy IMAGE is conventionally a
                # host tensor and many downstream nodes assume it. "same" keeps
                # NVDEC frames on the GPU.
                "output_device": (["cpu", "same"],),
                "resize_width": ("INT", {"default": 0, "min": 0}),
                "resize_height": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "fps", "width", "height")
    FUNCTION = "load"
    CATEGORY = "Nelux"

    @classmethod
    def VALIDATE_INPUTS(cls, input_path, **kwargs):
        return _validate_path(input_path)

    @classmethod
    def IS_CHANGED(cls, input_path="", **kwargs):
        return _file_signature(input_path)

    def load(self, input_path, start_frame, frame_count, step, decode_accelerator,
             output_device, resize_width, resize_height):
        nelux = _import_nelux()
        resize = (resize_width, resize_height) if resize_width and resize_height else None
        accelerator = _resolve_accelerator(input_path, decode_accelerator)
        with nelux.VideoReader(
            input_path, force_8bit=True,
            decode_accelerator=accelerator, resize=resize,
        ) as reader:
            start, end, count = _load_frames_plan(
                int(reader.total_frames), start_frame, frame_count, step
            )
            # nelux's batch API (get_batch_range) would skip the per-frame Python
            # round trip and seek instead of decoding the prefix -- ~1.9x on a
            # strided CPU read starting late in the file -- but on the CPU
            # backend (NevermindNilas/Nelux#58) it converts YUV->RGB with
            # BT.601 regardless of the
            # stream's declared colour space, so a bt709 clip comes out visibly
            # wrong (byte-exact against `ffmpeg -vf scale=in_color_matrix=bt601`,
            # max error 40/255 vs plain ffmpeg). Iteration is byte-exact against
            # ffmpeg, so the node iterates and eats the cost.
            frames = [
                _to_comfy_image(frame)
                for i, frame in enumerate(_iter_range(reader, start, end, accelerator))
                if i % step == 0
            ][:count]
            if not frames:
                raise RuntimeError(f"Nelux decoded no frames from {input_path}")
            images = torch.stack(frames)
            if output_device == "cpu":
                images = images.cpu()
            return images, float(reader.fps), int(reader.width), int(reader.height)


class NeluxEncodeFrames:
    """Direct IMAGE batch -> file encode (no VIDEO wrapper, no audio)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "nelux_comfy_output.mp4"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "codec": (_CODECS,),
                "encoder_engine": (_ENGINES,),
                "preset": (_PRESETS,),
                "cq": ("INT", {"default": 20, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "encode"
    CATEGORY = "Nelux"
    OUTPUT_NODE = True

    def encode(self, images, output_path, fps, codec, encoder_engine, preset, cq):
        path = _resolve_output_path(output_path)
        _nelux_encode(images, fps, path, codec=codec, preset=preset, cq=cq,
                      encoder_engine=encoder_engine)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "NeluxLoadVideo": NeluxLoadVideo,
    "NeluxSaveVideo": NeluxSaveVideo,
    "NeluxVideoInfo": NeluxVideoInfo,
    "NeluxLoadFrames": NeluxLoadFrames,
    "NeluxEncodeFrames": NeluxEncodeFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeluxLoadVideo": "Nelux Load Video",
    "NeluxSaveVideo": "Nelux Save Video",
    "NeluxVideoInfo": "Nelux Video Info",
    "NeluxLoadFrames": "Nelux Load Frames",
    "NeluxEncodeFrames": "Nelux Encode Frames",
}
