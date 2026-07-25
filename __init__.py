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

Every decode node defaults to ``decode_accelerator="auto"``, which probes for a
hardware decoder once per codec and falls back to the CPU, so the pack works
unchanged on a machine with no GPU.

Nelux must be importable in Comfy's Python (torch ABI must match the wheel). On
Windows the FFmpeg DLLs are located via NELUX_FFMPEG_DLL_DIR / FFMPEG_DLL_DIR;
set one of those env vars for your install.
"""

from __future__ import annotations

import logging
import os
import tempfile
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
# --------------------------------------------------------------------------- #
_DLL_HANDLES = []


def _add_dll_dir(path: str) -> None:
    if os.name == "nt" and path and os.path.isdir(path):
        _DLL_HANDLES.append(os.add_dll_directory(path))


def _import_nelux():
    for path in (os.environ.get("NELUX_FFMPEG_DLL_DIR"), os.environ.get("FFMPEG_DLL_DIR")):
        _add_dll_dir(path)
    try:
        import nelux
    except ImportError as exc:
        raise ImportError(
            "Nelux failed to import in Comfy's Python. Install a Nelux wheel built "
            "for this Comfy torch/Python combo; Comfy Desktop's bundled torch may "
            "not match the published Nelux wheel ABI. On Windows, also set "
            "NELUX_FFMPEG_DLL_DIR to the directory holding the FFmpeg DLLs."
        ) from exc
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
# nelux accepts exactly 'cpu', 'nvdec' and 'qsv' -- there is no 'auto', so the
# node resolves it. "auto" must never hard-fail on a machine with no NVIDIA GPU,
# which is what a plain default of "nvdec" would do.
# --------------------------------------------------------------------------- #
_ACCELERATORS = ["auto", "nvdec", "qsv", "cpu"]
_ACCEL_CACHE: dict[str, str] = {}


def _accelerator_works(nelux, path: str, accelerator: str) -> bool:
    try:
        with nelux.VideoReader(path, decode_accelerator=accelerator):
            return True
    except Exception:
        return False


def _resolve_accelerator(path: str, choice: str) -> str:
    """Map the node's accelerator choice onto one nelux accepts. "auto" probes
    for a working hardware decoder once per (codec, choice) and remembers the
    answer; anything else is passed through so an explicit request still raises
    rather than silently downgrading."""
    name = (choice or "auto").strip().lower()
    if name != "auto":
        return name
    nelux = _import_nelux()
    try:
        key = str(_probe(path).get("codec", "?"))
    except Exception:
        key = "?"
    cached = _ACCEL_CACHE.get(key)
    if cached is not None:
        return cached
    for candidate in ("nvdec", "qsv"):
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

# Comfy's VideoCodec values (and "auto") -> (preferred hardware, software fallback).
_CODEC_FALLBACKS = {
    "auto": ("h264_nvenc", "libx264"),
    "h264": ("h264_nvenc", "libx264"),
    "avc1": ("h264_nvenc", "libx264"),
    "h265": ("hevc_nvenc", "libx265"),
    "hevc": ("hevc_nvenc", "libx265"),
    "av1": ("av1_nvenc", "libsvtav1"),
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
    if encoder in _X26X_CODECS:
        return _X26X_BY_RUNG[rung]
    return None


def _available_nvenc(nelux) -> set[str]:
    try:
        return {e.get("name") for e in nelux.get_nvenc_encoders()}
    except Exception:
        return set()


def _resolve_encoder(nelux, codec: str) -> str:
    """Map a codec choice to a concrete encoder name. Generic names ("auto",
    "h264", "hevc", "av1") prefer NVENC and fall back to software when the
    hardware encoder is absent. An explicit NVENC name with no NVENC present is
    an error rather than a silent downgrade."""
    name = (codec or "auto").strip().lower()
    pair = _CODEC_FALLBACKS.get(name)
    if pair is None:
        if name in _NVENC_CODECS and name not in _available_nvenc(nelux):
            raise RuntimeError(
                f"Nelux: {name} was requested but no NVENC encoder is available "
                f"in this FFmpeg/driver combination. Use 'auto' to fall back to a "
                f"software encoder."
            )
        return name
    hardware, software = pair
    return hardware if hardware in _available_nvenc(nelux) else software


def _encoder_kwargs(nelux, codec: str, width: int, height: int, fps: float,
                    preset, cq: int) -> dict:
    encoder = _resolve_encoder(nelux, codec)
    kwargs = {
        "codec": encoder,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "cq": int(cq),
    }
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
            nelux, codec, int(reader.width), int(reader.height), fps, preset, cq
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
        nelux, codec, int(images.shape[2]), int(images.shape[1]), fps, preset, cq
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
_CODECS = [
    "auto", "h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "libx265", "libsvtav1",
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

    def save(self, video, filename_prefix, codec, container, preset, cq, audio):
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
                codec=codec, preset=preset, cq=cq,
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
                codec=codec, preset=preset, cq=cq,
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
                "preset": (_PRESETS,),
                "cq": ("INT", {"default": 20, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "encode"
    CATEGORY = "Nelux"
    OUTPUT_NODE = True

    def encode(self, images, output_path, fps, codec, preset, cq):
        path = _resolve_output_path(output_path)
        _nelux_encode(images, fps, path, codec=codec, preset=preset, cq=cq)
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
