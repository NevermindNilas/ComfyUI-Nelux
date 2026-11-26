"""Unit tests for the ComfyUI Nelux nodes.

These exec the node module directly and only touch its Comfy-independent,
PyAV-independent, nelux-independent surface, so they run in any environment.
Comfy's video API is stubbed where a test needs the VIDEO class; the real
nelux/PyAV/Comfy integration paths are exercised by the e2e scripts.
"""

import contextlib
import importlib.util
import math
import sys
import types
import wave
from pathlib import Path

import pytest
import torch


def _load_nodes():
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("comfyui_nelux_nodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeNelux:
    """Stands in for the nelux module in encoder-resolution tests.

    Availability is decided the way the real code decides it -- by trying to
    open a VideoEncoder -- so `working` is the set of encoders this fake machine
    can actually run."""

    def __init__(self, working=()):
        self.working = set(working)
        self.probed = []

    def VideoEncoder(self, path, codec=None, **kwargs):
        self.probed.append(codec)
        if codec not in self.working:
            raise RuntimeError(f"Failed to open video codec: {codec}")
        return contextlib.nullcontext()


_NVENC_ONLY = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
_QSV_ONLY = ("h264_qsv", "hevc_qsv", "av1_qsv")


@pytest.fixture
def encoders():
    """Factory for a fake machine. Each call re-execs the node module so the
    probe cache starts empty -- one machine's answers must not leak into the
    next one's."""

    def build(*working):
        return _load_nodes(), _FakeNelux(working)

    return build


@pytest.fixture
def comfy_video_class(monkeypatch):
    """Install a minimal fake comfy_api.latest so the Nelux VIDEO class builds."""
    latest = types.ModuleType("comfy_api.latest")
    package = types.ModuleType("comfy_api")

    class _VideoBase:
        pass

    class _VideoComponents:
        def __init__(self, images, frame_rate, audio=None, metadata=None):
            self.images, self.frame_rate, self.audio = images, frame_rate, audio

    latest.Input = type("Input", (), {"Video": _VideoBase})
    latest.InputImpl = type("InputImpl", (), {})
    latest.Types = type("Types", (), {"VideoComponents": _VideoComponents})
    package.latest = latest

    monkeypatch.setitem(sys.modules, "comfy_api", package)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    return _load_nodes()._nelux_video_class()


def _video(cls, start_time=0.0, duration=0.0, clip_seconds=30.0, fps=30.0):
    """A VIDEO whose header props are pre-cached, so no file is ever opened."""
    video = cls("clip.mp4", "cpu", start_time=start_time, duration=duration)
    video._props_cache = (1920, 1080, fps, int(clip_seconds * fps), clip_seconds)
    return video


# --------------------------------------------------------------------------- #
# FFmpeg bootstrap
#
# nelux DELAY-loads FFmpeg, so `import nelux` succeeds with no FFmpeg present
# and the failure only lands later, inside the first decode, as an uncatchable
# process abort. The DLLs therefore have to be checked directly, up front.
# --------------------------------------------------------------------------- #
@pytest.fixture
def win_nodes(monkeypatch):
    """Node module that believes it is on Windows, with a clean environment."""
    nodes = _load_nodes()
    monkeypatch.setattr(nodes, "_is_windows", lambda: True)
    monkeypatch.setattr(nodes, "_add_dll_dir", lambda path: None)
    for var in ("NELUX_FFMPEG_DLL_DIR", "FFMPEG_DLL_DIR",
                "NELUX_FFMPEG_CACHE_DIR", "NELUX_NO_AUTO_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    return nodes


def _loadable(nodes, monkeypatch, *names):
    monkeypatch.setattr(nodes, "_can_load_dll", lambda name: name in set(names))


def test_missing_ffmpeg_lists_every_unsatisfied_dependency(win_nodes, monkeypatch):
    _loadable(win_nodes, monkeypatch)
    assert win_nodes._missing_ffmpeg_dlls() == [
        "avcodec-63.dll", "avformat-63.dll", "avutil-61.dll",
        "swscale-10.dll", "swresample-7.dll",
    ]


def test_either_ffmpeg_generation_satisfies_the_requirement(win_nodes, monkeypatch):
    # A wheel linked against the older sonames must not trigger a download.
    _loadable(win_nodes, monkeypatch, "avcodec-62.dll", "avformat-62.dll",
              "avutil-60.dll", "swscale-9.dll", "swresample-6.dll")
    assert win_nodes._missing_ffmpeg_dlls() == []


def test_nothing_is_downloaded_when_ffmpeg_is_already_there(win_nodes, monkeypatch):
    _loadable(win_nodes, monkeypatch, "avcodec-63.dll", "avformat-63.dll",
              "avutil-61.dll", "swscale-10.dll", "swresample-7.dll")
    monkeypatch.setattr(win_nodes, "_download_ffmpeg", lambda dest: pytest.fail(
        "downloaded FFmpeg despite it already being loadable"))

    win_nodes._ensure_ffmpeg()


def test_ffmpeg_is_downloaded_once_when_missing(win_nodes, monkeypatch, tmp_path):
    calls = []
    resolved = {"ok": False}
    monkeypatch.setattr(win_nodes, "_can_load_dll", lambda name: resolved["ok"])

    def fake_download(destination):
        calls.append(destination)
        resolved["ok"] = True
        return str(tmp_path / "bin")

    monkeypatch.setattr(win_nodes, "_download_ffmpeg", fake_download)
    monkeypatch.setenv("NELUX_FFMPEG_CACHE_DIR", str(tmp_path))

    win_nodes._ensure_ffmpeg()
    win_nodes._ensure_ffmpeg()  # already resolvable now

    assert calls == [str(tmp_path)]


def test_download_can_be_turned_off(win_nodes, monkeypatch):
    _loadable(win_nodes, monkeypatch)
    monkeypatch.setenv("NELUX_NO_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(win_nodes, "_download_ffmpeg", lambda dest: pytest.fail(
        "downloaded FFmpeg despite NELUX_NO_AUTO_DOWNLOAD"))

    with pytest.raises(ImportError, match="Automatic download is disabled"):
        win_nodes._ensure_ffmpeg()


def test_a_failed_download_explains_itself(win_nodes, monkeypatch):
    _loadable(win_nodes, monkeypatch)

    def boom(destination):
        raise OSError("no route to host")

    monkeypatch.setattr(win_nodes, "_download_ffmpeg", boom)

    with pytest.raises(ImportError, match="no route to host"):
        win_nodes._ensure_ffmpeg()


def test_a_download_that_does_not_satisfy_the_sonames_is_reported(
    win_nodes, monkeypatch, tmp_path
):
    # An FFmpeg build of the wrong generation would otherwise leave the nodes to
    # abort inside the first decode instead of saying what went wrong.
    _loadable(win_nodes, monkeypatch)
    monkeypatch.setattr(win_nodes, "_download_ffmpeg", lambda dest: str(tmp_path))

    with pytest.raises(ImportError, match="does not match the sonames"):
        win_nodes._ensure_ffmpeg()


def test_non_windows_never_checks_or_downloads(monkeypatch):
    nodes = _load_nodes()
    monkeypatch.setattr(nodes, "_is_windows", lambda: False)

    assert nodes._missing_ffmpeg_dlls() == []
    assert nodes._auto_download_enabled() is False


def test_cache_dir_is_outside_the_node_directory(monkeypatch, tmp_path):
    # A checkout can be read-only or git-managed; downloads must not land in it.
    nodes = _load_nodes()
    monkeypatch.delenv("NELUX_FFMPEG_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    cache = Path(nodes._ffmpeg_cache_dir())

    assert tmp_path in cache.parents
    assert Path(nodes.__file__).parent not in cache.parents


def test_cache_dir_override_wins(monkeypatch, tmp_path):
    nodes = _load_nodes()
    monkeypatch.setenv("NELUX_FFMPEG_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert nodes._ffmpeg_cache_dir() == str(tmp_path / "elsewhere")


def test_download_flattens_archive_paths(monkeypatch, tmp_path):
    """Entries are taken by basename, so a crafted path cannot escape the
    destination, and the DLLs land in one flat directory the loader can search."""
    import io
    import zipfile

    nodes = _load_nodes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("ffmpeg-master/bin/avcodec-63.dll", b"codec")
        zf.writestr("ffmpeg-master/bin/avutil-61.dll", b"util")
        zf.writestr("ffmpeg-master/bin/../../evil/bin/pwned.dll", b"evil")
        zf.writestr("ffmpeg-master/doc/readme.txt", b"not a dll")
        zf.writestr("ffmpeg-master/lib/avcodec.lib", b"import lib")
    payload = buffer.getvalue()

    monkeypatch.setattr(
        __import__("urllib.request", fromlist=["request"]), "urlopen",
        lambda request, timeout=None: io.BytesIO(payload),
    )

    destination = tmp_path / "ffmpeg"
    bin_dir = Path(nodes._download_ffmpeg(str(destination)))

    names = sorted(p.name for p in bin_dir.iterdir())
    assert names == ["avcodec-63.dll", "avutil-61.dll", "pwned.dll"]
    assert not (tmp_path / "evil").exists()  # never escaped
    assert bin_dir == destination / "bin"


def test_an_archive_without_ffmpeg_is_rejected(monkeypatch, tmp_path):
    import io
    import zipfile

    nodes = _load_nodes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("something/bin/unrelated.dll", b"nope")
    payload = buffer.getvalue()
    monkeypatch.setattr(
        __import__("urllib.request", fromlist=["request"]), "urlopen",
        lambda request, timeout=None: io.BytesIO(payload),
    )

    with pytest.raises(RuntimeError, match="no avcodec DLL"):
        nodes._download_ffmpeg(str(tmp_path / "ffmpeg"))
    assert not (tmp_path / "ffmpeg").exists()  # nothing half-written left behind


# --------------------------------------------------------------------------- #
# Node contract
# --------------------------------------------------------------------------- #
def test_comfy_image_roundtrip_shape_dtype():
    nodes = _load_nodes()
    frame = torch.tensor([[[0, 127, 255]]], dtype=torch.uint8)

    image = nodes._to_comfy_image(frame)
    back = nodes._to_nelux_frame(image)

    assert image.shape == (1, 1, 3)
    assert image.dtype == torch.float32
    assert torch.equal(back, frame)


def test_to_comfy_image_never_scales_the_source_buffer():
    # A float32 frame would alias through .to(float32); scaling in place would
    # corrupt the decoder's own buffer.
    nodes = _load_nodes()
    frame = torch.full((1, 1, 3), 255.0, dtype=torch.float32)

    image = nodes._to_comfy_image(frame)

    assert torch.equal(frame, torch.full((1, 1, 3), 255.0))
    assert torch.equal(image, torch.ones(1, 1, 3))


def test_node_mappings_exist():
    nodes = _load_nodes()

    expected = {
        "NeluxLoadVideo",
        "NeluxSaveVideo",
        "NeluxVideoInfo",
        "NeluxLoadFrames",
        "NeluxEncodeFrames",
    }
    assert expected <= set(nodes.NODE_CLASS_MAPPINGS)
    assert expected <= set(nodes.NODE_DISPLAY_NAME_MAPPINGS)


def test_video_nodes_use_video_type():
    nodes = _load_nodes()
    assert nodes.NeluxLoadVideo.RETURN_TYPES == ("VIDEO",)
    assert nodes.NeluxSaveVideo.INPUT_TYPES()["required"]["video"][0] == "VIDEO"
    assert nodes.NeluxSaveVideo.OUTPUT_NODE is True


def test_load_frames_output_device_defaults_to_cpu():
    # First combo entry is the Comfy default; a host tensor is the safe default
    # so downstream nodes that assume CPU IMAGE tensors keep working.
    nodes = _load_nodes()
    devices = nodes.NeluxLoadFrames.INPUT_TYPES()["required"]["output_device"][0]
    assert devices[0] == "cpu"


def test_validate_inputs_rejects_missing_and_empty(tmp_path):
    nodes = _load_nodes()

    assert nodes.NeluxVideoInfo.VALIDATE_INPUTS("") != True  # noqa: E712
    assert isinstance(nodes.NeluxVideoInfo.VALIDATE_INPUTS(str(tmp_path / "nope.mp4")), str)

    real = tmp_path / "clip.mp4"
    real.write_bytes(b"not really a video, but a real file")
    assert nodes.NeluxVideoInfo.VALIDATE_INPUTS(str(real)) is True


def test_is_changed_tracks_mtime(tmp_path):
    nodes = _load_nodes()

    assert math.isnan(nodes.NeluxVideoInfo.IS_CHANGED(str(tmp_path / "missing.mp4")))

    real = tmp_path / "clip.mp4"
    real.write_bytes(b"x")
    assert nodes.NeluxVideoInfo.IS_CHANGED(str(real)) == real.stat().st_mtime


def test_video_info_return_contract():
    nodes = _load_nodes()
    info = nodes.NeluxVideoInfo
    assert len(info.RETURN_TYPES) == len(info.RETURN_NAMES)
    assert info.RETURN_NAMES[0] == "width"


# --------------------------------------------------------------------------- #
# Preset normalization
#
# nelux forwards a string preset to av_dict_set("preset", ...) verbatim, so a
# name from the wrong family makes avcodec_open2 fail outright:
#   x264 [error]: invalid preset 'p4'
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "encoder,preset,expected",
    [
        ("libx264", "p4", "medium"),
        ("libx264", "p1", "ultrafast"),
        ("libx265", "p7", "veryslow"),
        ("h264_nvenc", "medium", "p4"),
        ("h264_nvenc", "veryslow", "p7"),
        ("h264_nvenc", "p4", "p4"),
        ("hevc_nvenc", "fastest", "p1"),
    ],
)
def test_preset_is_translated_into_the_encoder_family(encoder, preset, expected):
    nodes = _load_nodes()
    assert nodes._resolve_preset(encoder, preset) == expected


@pytest.mark.parametrize("preset", ["auto", "", None, "  "])
def test_preset_auto_defers_to_the_encoder(preset):
    nodes = _load_nodes()
    assert nodes._resolve_preset("libx264", preset) is None


def test_preset_unknown_name_passes_through():
    # e.g. NVENC's "lossless", or libsvtav1's numeric presets.
    nodes = _load_nodes()
    assert nodes._resolve_preset("h264_nvenc", "lossless") == "lossless"
    assert nodes._resolve_preset("libsvtav1", "8") == "8"


def test_preset_dropped_for_encoders_with_no_known_ladder():
    nodes = _load_nodes()
    assert nodes._resolve_preset("libsvtav1", "medium") is None


# --------------------------------------------------------------------------- #
# Encoder resolution
#
# Availability must come from actually opening the encoder: nelux's
# get_nvenc_encoders() only reports what FFmpeg was compiled with, and this
# FFmpeg build lists NVENC/QSV/AMF/VAAPI encoders on every platform.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "codec,nvenc_box,qsv_box,cpu_box",
    [
        ("auto", "h264_nvenc", "h264_qsv", "libx264"),
        ("h264", "h264_nvenc", "h264_qsv", "libx264"),
        ("hevc", "hevc_nvenc", "hevc_qsv", "libx265"),
        ("h265", "hevc_nvenc", "hevc_qsv", "libx265"),
        ("av1", "av1_nvenc", "av1_qsv", "libsvtav1"),
    ],
)
def test_auto_engine_prefers_nvenc_then_qsv_then_cpu(
    encoders, codec, nvenc_box, qsv_box, cpu_box
):
    nodes, nvidia = encoders(*_NVENC_ONLY, *_QSV_ONLY)
    assert nodes._resolve_encoder(nvidia, codec) == nvenc_box

    nodes, intel = encoders(*_QSV_ONLY)
    assert nodes._resolve_encoder(intel, codec) == qsv_box

    nodes, bare = encoders()
    assert nodes._resolve_encoder(bare, codec) == cpu_box


@pytest.mark.parametrize(
    "engine,expected",
    [("nvenc", "h264_nvenc"), ("qsv", "h264_qsv"), ("cpu", "libx264")],
)
def test_explicit_engine_is_honoured(encoders, engine, expected):
    nodes, machine = encoders(*_NVENC_ONLY, *_QSV_ONLY)
    assert nodes._resolve_encoder(machine, "h264", engine) == expected


def test_explicit_engine_raises_rather_than_downgrading(encoders):
    # Asking for QSV on a machine with none must not quietly hand back libx264.
    nodes, nvidia = encoders(*_NVENC_ONLY)
    with pytest.raises(RuntimeError, match="cannot encode h264"):
        nodes._resolve_encoder(nvidia, "h264", "qsv")


def test_unknown_engine_is_rejected(encoders):
    nodes, machine = encoders(*_NVENC_ONLY)
    with pytest.raises(RuntimeError, match="unknown encoder_engine"):
        nodes._resolve_encoder(machine, "h264", "vaapi")


def test_cpu_engine_never_probes_hardware(encoders):
    nodes, machine = encoders()
    assert nodes._resolve_encoder(machine, "hevc", "cpu") == "libx265"
    assert machine.probed == []


def test_exact_encoder_name_wins_over_the_engine(encoders):
    nodes, machine = encoders(*_NVENC_ONLY, *_QSV_ONLY)
    assert nodes._resolve_encoder(machine, "libx264", "nvenc") == "libx264"
    assert nodes._resolve_encoder(machine, "hevc_qsv", "nvenc") == "hevc_qsv"


def test_exact_hardware_name_without_the_hardware_raises(encoders):
    # av1_nvenc is in every FFmpeg build but needs Ada or newer -- it fails to
    # open on, say, an RTX 3090, and a build-time name list cannot tell.
    nodes, ampere = encoders("h264_nvenc", "hevc_nvenc")
    with pytest.raises(RuntimeError, match="cannot open it"):
        nodes._resolve_encoder(ampere, "av1_nvenc")


def test_auto_engine_skips_a_codec_the_gpu_cannot_do(encoders):
    # Same Ampere box: h264 goes to NVENC, but av1 has to fall past it.
    nodes, ampere = encoders("h264_nvenc", "hevc_nvenc")
    assert nodes._resolve_encoder(ampere, "h264") == "h264_nvenc"
    assert nodes._resolve_encoder(ampere, "av1") == "libsvtav1"


def test_each_encoder_is_probed_only_once(encoders):
    nodes, machine = encoders(*_NVENC_ONLY)
    for _ in range(5):
        nodes._resolve_encoder(machine, "h264")
    assert machine.probed == ["h264_nvenc"]


def test_encoder_kwargs_omit_preset_when_auto(encoders):
    nodes, bare = encoders()
    kwargs = nodes._encoder_kwargs(bare, "auto", 64, 32, 24.0, "auto", 20)
    assert kwargs == {"codec": "libx264", "width": 64, "height": 32, "fps": 24.0, "cq": 20}


def test_encoder_kwargs_translate_preset_for_the_resolved_encoder(encoders):
    nodes, bare = encoders()
    nodes_hw, nvidia = encoders(*_NVENC_ONLY)
    assert nodes._encoder_kwargs(bare, "auto", 64, 32, 24.0, "p4", 20)["preset"] == "medium"
    assert nodes_hw._encoder_kwargs(nvidia, "auto", 64, 32, 24.0, "p4", 20)["preset"] == "p4"


# --------------------------------------------------------------------------- #
# QSV quality and presets
#
# nelux maps `cq` onto crf/qp for x26x, SVT-AV1, libaom and NVENC, but QSV
# matches none of those branches, so passing `cq` there is silently dropped --
# encoding the same clip with and without it gives byte-identical output.
# --------------------------------------------------------------------------- #
def test_qsv_quality_goes_through_global_quality(encoders):
    nodes, intel = encoders(*_QSV_ONLY)

    kwargs = nodes._encoder_kwargs(intel, "h264", 64, 32, 24.0, "auto", 23, engine="qsv")

    assert kwargs["codec"] == "h264_qsv"
    assert kwargs["options"] == {"global_quality": "23"}
    assert "cq" not in kwargs  # nelux would drop it on the floor


def test_non_qsv_encoders_still_use_cq(encoders):
    nodes, nvidia = encoders(*_NVENC_ONLY)
    kwargs = nodes._encoder_kwargs(nvidia, "h264", 64, 32, 24.0, "auto", 23, engine="nvenc")
    assert kwargs["cq"] == 23
    assert "options" not in kwargs


@pytest.mark.parametrize(
    "preset,expected",
    [("p1", "veryfast"), ("p4", "medium"), ("p7", "veryslow"), ("medium", "medium")],
)
def test_qsv_preset_ladder(preset, expected):
    # QSV has no ultrafast/superfast rung, so its ladder is one step slower than
    # x264's at the fast end.
    nodes = _load_nodes()
    assert nodes._resolve_preset("h264_qsv", preset) == expected


# --------------------------------------------------------------------------- #
# Raw PCM container guard
#
# The mp4 muxer takes a pcm_s16le stream, writes a zero-length stsd entry, and
# reports success -- the file then fails to demux ("invalid size 0 in stsd").
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["out.mkv", "out.mov", "OUT.MKV"])
def test_pcm_allowed_in_containers_that_hold_it(path):
    nodes = _load_nodes()
    assert nodes._pcm_container_error(path) is None


@pytest.mark.parametrize("path", ["out.mp4", "out.webm", "out"])
def test_pcm_rejected_in_containers_that_corrupt_it(path):
    nodes = _load_nodes()
    assert "cannot be muxed" in nodes._pcm_container_error(path)


def test_encode_rejects_pcm_passthrough_into_mp4():
    nodes = _load_nodes()
    images = torch.zeros(2, 8, 8, 3)
    with pytest.raises(ValueError, match="cannot be muxed"):
        nodes._nelux_encode(images, 24.0, "out.mp4", passthrough_source="audio.wav")


def test_encode_allows_compressed_passthrough_into_mp4(monkeypatch):
    # A file-backed source carries an already-compressed audio stream, so the
    # guard must not fire; stop before touching nelux.
    nodes = _load_nodes()
    monkeypatch.setattr(nodes, "_import_nelux", lambda: (_ for _ in ()).throw(RuntimeError("reached nelux")))
    with pytest.raises(RuntimeError, match="reached nelux"):
        nodes._nelux_encode(torch.zeros(2, 8, 8, 3), 24.0, "out.mp4", passthrough_source="src.mp4")


# --------------------------------------------------------------------------- #
# Frame-range math
# --------------------------------------------------------------------------- #
def test_frame_range_covers_whole_clip_by_default():
    nodes = _load_nodes()
    assert nodes._frame_range(30.0, 900, 0.0, 0.0) == (0, 900)


def test_frame_range_windows_on_start_and_duration():
    nodes = _load_nodes()
    assert nodes._frame_range(30.0, 900, 1.0, 2.0) == (30, 90)


def test_frame_range_clamps_past_the_end():
    nodes = _load_nodes()
    assert nodes._frame_range(30.0, 900, 100.0, 0.0) == (899, 900)
    assert nodes._frame_range(30.0, 900, 29.0, 10.0) == (870, 900)


def test_frame_range_reads_to_eof_when_the_header_has_no_count():
    nodes = _load_nodes()
    start, end = nodes._frame_range(30.0, 0, 1.0, 0.0)
    assert (start, end) == (30, 2**31 - 1)


# --------------------------------------------------------------------------- #
# Trim composition
# --------------------------------------------------------------------------- #
def test_as_trimmed_on_an_untrimmed_video_stays_unbounded(comfy_video_class):
    video = _video(comfy_video_class)
    child = video.as_trimmed(start_time=5.0)
    assert (child._start_time, child._duration) == (5.0, 0.0)


def test_as_trimmed_keeps_the_parent_window_when_no_duration_is_given(comfy_video_class):
    # A 10s window re-trimmed 2s in has 8s left -- not the 28s remaining in the file.
    video = _video(comfy_video_class, duration=10.0)
    child = video.as_trimmed(start_time=2.0)
    assert (child._start_time, child._duration) == (2.0, 8.0)


def test_as_trimmed_clamps_a_child_that_overruns_its_parent(comfy_video_class):
    video = _video(comfy_video_class, start_time=1.0, duration=10.0)
    child = video.as_trimmed(start_time=2.0, duration=20.0)
    assert (child._start_time, child._duration) == (3.0, 8.0)


def test_as_trimmed_clamps_against_the_end_of_the_file(comfy_video_class):
    video = _video(comfy_video_class, clip_seconds=30.0)
    child = video.as_trimmed(start_time=25.0, duration=100.0)
    assert child._duration == 5.0


def test_as_trimmed_strict_duration_rejects_an_impossible_window(comfy_video_class):
    video = _video(comfy_video_class, duration=10.0)
    assert video.as_trimmed(start_time=5.0, duration=20.0, strict_duration=True) is None
    assert video.as_trimmed(start_time=5.0, duration=5.0, strict_duration=True) is not None


def test_get_duration_and_frame_count_respect_the_window(comfy_video_class):
    video = _video(comfy_video_class, start_time=2.0, duration=3.0, fps=30.0)
    assert video.get_duration() == 3.0
    assert video.get_frame_count() == 90


def test_get_frame_count_is_the_header_count_when_untrimmed(comfy_video_class):
    video = _video(comfy_video_class, clip_seconds=30.0, fps=30.0)
    assert video.get_frame_count() == 900


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def test_write_waveform_wav_roundtrip(tmp_path):
    nodes = _load_nodes()
    sample_rate = 8000
    # Two channels, 100 samples, values in [-1, 1].
    t = torch.linspace(-1.0, 1.0, steps=100)
    waveform = torch.stack([t, -t]).unsqueeze(0)  # [1, 2, 100]
    audio = {"waveform": waveform, "sample_rate": sample_rate}

    wav_path = tmp_path / "out.wav"
    assert nodes._write_waveform_wav(audio, str(wav_path)) is True

    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == sample_rate
        assert w.getsampwidth() == 2
        assert w.getnframes() == 100


def test_write_waveform_wav_empty_is_noop(tmp_path):
    nodes = _load_nodes()
    assert nodes._write_waveform_wav(None, str(tmp_path / "x.wav")) is False
    assert nodes._write_waveform_wav({}, str(tmp_path / "x.wav")) is False


# --------------------------------------------------------------------------- #
# Decode accelerator resolution
#
# nelux accepts only 'cpu' / 'nvdec' / 'qsv'. "auto" is the node's own concept
# and must degrade to CPU rather than hard-failing on a machine with no GPU.
# --------------------------------------------------------------------------- #
class _FakeReaderNelux:
    """Fake nelux whose VideoReader only opens for `working` accelerators."""

    def __init__(self, working=(), probe_codec="h264"):
        self.working = set(working)
        self.opened = []
        self._probe_codec = probe_codec

    def probe(self, path):
        return {"codec": self._probe_codec}

    def VideoReader(self, path, decode_accelerator="cpu", **kwargs):
        self.opened.append(decode_accelerator)
        if decode_accelerator not in self.working:
            raise RuntimeError(f"no {decode_accelerator}")
        return contextlib.nullcontext()


@pytest.fixture
def accel_nodes(monkeypatch):
    """Node module with a fresh (empty) accelerator cache."""
    nodes = _load_nodes()
    monkeypatch.setattr(nodes, "_ACCEL_CACHE", {})
    return nodes


@pytest.mark.parametrize(
    "working,expected",
    [
        (("nvdec", "qsv", "cpu"), "nvdec"),  # NVIDIA box: prefer NVDEC
        (("qsv", "cpu"), "qsv"),             # Intel box: fall through to QSV
        (("cpu",), "cpu"),                   # no hardware decoder at all
    ],
)
def test_auto_accelerator_picks_the_best_working_backend(
    accel_nodes, monkeypatch, working, expected
):
    fake = _FakeReaderNelux(working)
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: fake)
    assert accel_nodes._resolve_accelerator("clip.mp4", "auto") == expected


def test_auto_accelerator_never_raises_without_hardware(accel_nodes, monkeypatch):
    # A default of "nvdec" would make every node fail outright on a CPU-only box.
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: _FakeReaderNelux(()))
    assert accel_nodes._resolve_accelerator("clip.mp4", "auto") == "cpu"


def test_auto_accelerator_probes_once_per_codec(accel_nodes, monkeypatch):
    fake = _FakeReaderNelux(("nvdec", "cpu"))
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: fake)

    accel_nodes._resolve_accelerator("a.mp4", "auto")
    probes_after_first = len(fake.opened)
    accel_nodes._resolve_accelerator("b.mp4", "auto")

    assert len(fake.opened) == probes_after_first


@pytest.mark.parametrize("choice", ["nvdec", "qsv", "cpu"])
def test_explicit_accelerator_is_honoured(accel_nodes, monkeypatch, choice):
    # An explicit request is never downgraded to something else.
    fake = _FakeReaderNelux(("nvdec", "qsv", "cpu"))
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: fake)

    assert accel_nodes._resolve_accelerator("clip.mp4", choice) == choice


def test_explicit_cpu_is_never_probed(accel_nodes, monkeypatch):
    fake = _FakeReaderNelux(())
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: fake)

    assert accel_nodes._resolve_accelerator("clip.mp4", "cpu") == "cpu"
    assert fake.opened == []


@pytest.mark.parametrize("choice", ["nvdec", "qsv"])
def test_explicit_hardware_accelerator_raises_rather_than_downgrading(
    accel_nodes, monkeypatch, choice
):
    # nelux's own message for an unsupported backend is "Unknown
    # decode_accelerator: qsv" -- true but unhelpful, since it does not say
    # whether the build or the machine is at fault.
    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: _FakeReaderNelux(("cpu",)))

    with pytest.raises(RuntimeError, match="cannot decode this file"):
        accel_nodes._resolve_accelerator("clip.mp4", choice)


def test_accelerator_probe_matches_how_the_nodes_decode(accel_nodes, monkeypatch):
    # force_8bit changes which pixel formats a backend must handle, so probing
    # without it could bless a backend that then fails on the real open.
    seen = {}

    class _Recorder(_FakeReaderNelux):
        def VideoReader(self, path, decode_accelerator="cpu", **kwargs):
            seen[decode_accelerator] = kwargs
            return super().VideoReader(path, decode_accelerator, **kwargs)

    monkeypatch.setattr(accel_nodes, "_import_nelux", lambda: _Recorder(("nvdec",)))
    accel_nodes._resolve_accelerator("clip.mp4", "auto")

    assert seen["nvdec"].get("force_8bit") is True


def test_auto_is_the_default_accelerator_in_every_node():
    nodes = _load_nodes()
    for node in (nodes.NeluxLoadVideo, nodes.NeluxLoadFrames):
        options = node.INPUT_TYPES()["required"]["decode_accelerator"][0]
        assert options[0] == "auto", node.__name__


# --------------------------------------------------------------------------- #
# Trimmed iteration
#
# NevermindNilas/Nelux#57: on nvdec, set_range(start, end) + iterate yields
# absolute frames keyframe+start. The node decodes from 0 and drops the prefix.
# --------------------------------------------------------------------------- #
class _FakeReader:
    """Records set_range calls and yields its range as integers."""

    def __init__(self, total=1000):
        self.total = total
        self.ranges = []
        self._range = (0, total)

    def set_range(self, start, end):
        self.ranges.append((start, end))
        self._range = (start, end)

    def __iter__(self):
        return iter(range(self._range[0], min(self._range[1], self.total)))


@pytest.mark.parametrize("accelerator", ["cpu", "qsv"])
def test_iter_range_seeks_directly_on_backends_that_seek_correctly(accelerator):
    nodes = _load_nodes()
    reader = _FakeReader()

    frames = list(nodes._iter_range(reader, 300, 303, accelerator))

    assert reader.ranges == [(300, 303)]
    assert frames == [300, 301, 302]


def test_iter_range_decodes_from_zero_on_nvdec():
    nodes = _load_nodes()
    reader = _FakeReader()

    frames = list(nodes._iter_range(reader, 300, 303, "nvdec"))

    assert reader.ranges == [(0, 303)]  # never seeks: nvdec would land at 550
    assert frames == [300, 301, 302]


def test_iter_range_still_seeks_on_nvdec_when_starting_at_zero():
    # start == 0 lands on keyframe 0, where the offset bug cannot bite.
    nodes = _load_nodes()
    reader = _FakeReader()

    frames = list(nodes._iter_range(reader, 0, 3, "nvdec"))

    assert reader.ranges == [(0, 3)]
    assert frames == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Frame-count clamping
# --------------------------------------------------------------------------- #
def test_load_frames_plan_covers_a_plain_request():
    nodes = _load_nodes()
    assert nodes._load_frames_plan(600, 0, 16, 1) == (0, 16, 16)


def test_load_frames_plan_end_covers_only_the_sampled_frames():
    nodes = _load_nodes()
    # 3 frames at step 4 from 100 are 100, 104, 108 -- so the range ends at 109.
    assert nodes._load_frames_plan(600, 100, 3, 4) == (100, 109, 3)


def test_load_frames_plan_clamps_a_request_that_overruns_the_file():
    nodes = _load_nodes()
    # 10 frames left, 64 asked for.
    assert nodes._load_frames_plan(600, 590, 64, 1) == (590, 600, 10)


def test_load_frames_plan_clamps_a_start_past_the_end():
    nodes = _load_nodes()
    start, end, count = nodes._load_frames_plan(600, 10_000, 4, 1)
    assert (start, end, count) == (599, 600, 1)


def test_load_frames_plan_clamps_strided_requests_by_stride():
    nodes = _load_nodes()
    # 40 frames left at step 8 is 5 samples, not 40.
    assert nodes._load_frames_plan(600, 560, 32, 8) == (560, 593, 5)


def test_load_frames_plan_trusts_the_request_when_the_header_has_no_count():
    nodes = _load_nodes()
    assert nodes._load_frames_plan(0, 100, 8, 2) == (100, 115, 8)


# --------------------------------------------------------------------------- #
# Output path routing
# --------------------------------------------------------------------------- #
def test_output_path_is_anchored_in_comfys_output_directory(monkeypatch, tmp_path):
    nodes = _load_nodes()
    monkeypatch.setattr(
        nodes, "_folder_paths",
        lambda: types.SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )

    resolved = nodes._resolve_output_path("clips/out.mp4")

    assert Path(resolved) == tmp_path / "clips" / "out.mp4"
    assert (tmp_path / "clips").is_dir()  # created ahead of the encode


def test_absolute_output_path_is_left_alone(monkeypatch, tmp_path):
    nodes = _load_nodes()
    monkeypatch.setattr(
        nodes, "_folder_paths",
        lambda: types.SimpleNamespace(get_output_directory=lambda: str(tmp_path / "nope")),
    )
    absolute = str(tmp_path / "elsewhere.mp4")

    assert nodes._resolve_output_path(absolute) == absolute


def test_output_path_falls_back_when_comfy_is_absent():
    # The module has to work headless, e.g. under these tests.
    nodes = _load_nodes()
    assert nodes._resolve_output_path("out.mp4") == "out.mp4"


def test_empty_output_path_is_rejected():
    nodes = _load_nodes()
    with pytest.raises(ValueError, match="output_path is empty"):
        nodes._resolve_output_path("  ")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_video_info_reports_every_probe_field(monkeypatch):
    nodes = _load_nodes()
    probed = {
        "width": 1920, "height": 1080, "fps": 30.0, "total_frames": 600,
        "duration": 20.0, "pixel_format": "yuv420p", "has_audio": True,
        "codec": "h264", "bit_depth": 8,
    }
    monkeypatch.setattr(nodes, "_probe", lambda path: probed)

    values = nodes.NeluxVideoInfo().probe("clip.mp4")

    assert len(values) == len(nodes.NeluxVideoInfo.RETURN_NAMES)
    assert dict(zip(nodes.NeluxVideoInfo.RETURN_NAMES, values)) == probed


def test_video_info_never_opens_a_decoder(monkeypatch):
    # A VideoReader open allocates frame buffers and starts worker threads;
    # nelux.probe parses the header only.
    nodes = _load_nodes()
    monkeypatch.setattr(
        nodes, "_import_nelux",
        lambda: (_ for _ in ()).throw(AssertionError("opened a decoder")),
    )
    monkeypatch.setattr(nodes, "_probe", lambda path: {
        "width": 1, "height": 1, "fps": 1.0, "total_frames": 1, "duration": 1.0,
        "pixel_format": "yuv420p", "has_audio": False, "codec": "h264", "bit_depth": 8,
    })

    assert nodes.NeluxVideoInfo().probe("clip.mp4")[0] == 1


def test_video_class_metadata_comes_from_probe(comfy_video_class, monkeypatch):
    # The class closes over its own module's globals, which is what _props()
    # resolves _probe through.
    monkeypatch.setitem(
        comfy_video_class.__init__.__globals__, "_probe",
        lambda path: {"width": 1920, "height": 1080, "fps": 30.0,
                      "total_frames": 900, "duration": 30.0},
    )

    video = comfy_video_class("clip.mp4", "auto")

    assert video.get_dimensions() == (1920, 1080)
    assert video.get_frame_count() == 900


def test_comfy_codec_name_unwraps_enum_values():
    nodes = _load_nodes()

    class _Codec:
        value = "H264"

    assert nodes._comfy_codec_name(_Codec()) == "h264"
    assert nodes._comfy_codec_name(None) == "auto"
    assert nodes._comfy_codec_name("AV1") == "av1"


# --------------------------------------------------------------------------- #
# Streaming CPU frame collection (prevents VRAM OOM during NVDEC decode)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("expected_count", [0, 3, 5, 8, 20])
def test_collect_comfy_images_cpu_handles_exact_under_over_and_unknown_counts(
    expected_count,
):
    nodes = _load_nodes()
    source = (torch.full((4, 6, 3), i, dtype=torch.uint8) for i in range(5))

    images = nodes._collect_comfy_images_cpu(source, expected_count=expected_count)

    assert images.shape == (5, 4, 6, 3)
    assert images.dtype == torch.float32
    assert images.device.type == "cpu"
    for i in range(5):
        assert torch.allclose(images[i], torch.full((4, 6, 3), i / 255.0))


def test_collect_comfy_images_cpu_returns_none_for_empty_source():
    nodes = _load_nodes()
    assert nodes._collect_comfy_images_cpu(iter(()), expected_count=10) is None
    assert nodes._collect_comfy_images_cpu(iter(()), expected_count=0) is None


def test_collect_comfy_images_cpu_does_not_retain_previous_frames():
    import weakref

    nodes = _load_nodes()
    refs = []
    alive_at_yield = []

    def source():
        for i in range(5):
            alive_at_yield.append(sum(1 for r in refs if r() is not None))
            frame = torch.full((4, 4, 3), i, dtype=torch.uint8)
            refs.append(weakref.ref(frame))
            yield frame
            del frame

    images = nodes._collect_comfy_images_cpu(source(), expected_count=5)

    assert images.shape == (5, 4, 4, 3)
    # Each frame is copied into the CPU buffer and released before the next is read.
    assert max(alive_at_yield) == 0


def test_decode_frames_nelux_streams_into_cpu_tensor(monkeypatch):
    nodes = _load_nodes()

    class _FakeVideoReader:
        def __init__(self, path, **kwargs):
            self.fps = 30.0
            self.width = 4
            self.height = 2
            self.total_frames = 4
            self._range = (0, 4)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_range(self, start, end):
            self._range = (start, end)

        def __iter__(self):
            for i in range(*self._range):
                yield torch.full((self.height, self.width, 3), i * 10, dtype=torch.uint8)

    fake_nelux = types.SimpleNamespace(VideoReader=_FakeVideoReader)
    monkeypatch.setattr(nodes, "_import_nelux", lambda: fake_nelux)
    monkeypatch.setattr(nodes, "_resolve_accelerator", lambda path, choice: "cpu")

    images, fps, width, height = nodes._decode_frames_nelux("clip.mp4", "auto")

    assert (fps, width, height) == (30.0, 4, 2)
    assert images.shape == (4, 2, 4, 3)
    assert images.device.type == "cpu"
    assert torch.allclose(images[3], torch.full((2, 4, 3), 30.0 / 255.0))


def test_load_frames_cpu_streams_strided_frames(monkeypatch):
    nodes = _load_nodes()

    class _FakeVideoReader:
        def __init__(self, path, **kwargs):
            self.fps = 24.0
            self.width = 2
            self.height = 2
            self.total_frames = 10
            self._range = (0, 10)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_range(self, start, end):
            self._range = (start, end)

        def __iter__(self):
            for i in range(*self._range):
                yield torch.full((2, 2, 3), i, dtype=torch.uint8)

    fake_nelux = types.SimpleNamespace(VideoReader=_FakeVideoReader)
    monkeypatch.setattr(nodes, "_import_nelux", lambda: fake_nelux)
    monkeypatch.setattr(nodes, "_resolve_accelerator", lambda path, choice: "cpu")

    images, fps, width, height = nodes.NeluxLoadFrames().load(
        "clip.mp4", start_frame=1, frame_count=3, step=2,
        decode_accelerator="auto", output_device="cpu",
        resize_width=0, resize_height=0,
    )

    assert (fps, width, height) == (24.0, 2, 2)
    assert images.shape == (3, 2, 2, 3)
    assert images.device.type == "cpu"
    # Sampled frames: 1, 3, 5.
    assert torch.allclose(images[0], torch.full((2, 2, 3), 1.0 / 255.0))
    assert torch.allclose(images[1], torch.full((2, 2, 3), 3.0 / 255.0))
    assert torch.allclose(images[2], torch.full((2, 2, 3), 5.0 / 255.0))


# --------------------------------------------------------------------------- #
# Component-backed audio AAC encoding for MP4
# --------------------------------------------------------------------------- #
def test_write_waveform_aac_empty_is_noop(tmp_path):
    nodes = _load_nodes()
    out = str(tmp_path / "x.m4a")
    assert nodes._write_waveform_aac(None, out) is False
    assert nodes._write_waveform_aac({}, out) is False
    assert nodes._write_waveform_aac({"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100}, out) is False
    assert nodes._write_waveform_aac({"waveform": torch.zeros(1, 2, 100), "sample_rate": 0}, out) is False


def _stub_save_env(nodes, monkeypatch, tmp_path):
    monkeypatch.setattr(
        nodes, "_folder_paths",
        lambda: types.SimpleNamespace(
            get_output_directory=lambda: str(tmp_path),
            get_save_image_path=lambda prefix, out: (str(tmp_path), "clip", 1, "", prefix),
        ),
    )
    monkeypatch.setattr(nodes, "_nelux_video_class", lambda: type("_DummyNeluxVideo", (), {}))


def _fake_component_video():
    audio = {"waveform": torch.zeros(1, 2, 1024), "sample_rate": 44100}
    components = types.SimpleNamespace(
        images=torch.zeros(2, 8, 8, 3),
        frame_rate=24.0,
        audio=audio,
    )
    return types.SimpleNamespace(get_components=lambda: components)


def test_save_video_encodes_component_audio_to_aac_for_mp4(monkeypatch, tmp_path):
    nodes = _load_nodes()
    _stub_save_env(nodes, monkeypatch, tmp_path)

    aac_paths = []
    encode_calls = []

    def fake_write_aac(audio, path):
        aac_paths.append(path)
        Path(path).write_bytes(b"fake-m4a")
        return True

    def fake_encode(images, fps, out_path, **kwargs):
        encode_calls.append((out_path, kwargs["passthrough_source"]))
        return out_path

    monkeypatch.setattr(nodes, "_write_waveform_aac", fake_write_aac)
    monkeypatch.setattr(nodes, "_nelux_encode", fake_encode)

    nodes.NeluxSaveVideo().save(
        _fake_component_video(), "out/clip", "auto", "auto", "mp4", "auto", 20, True
    )

    assert len(aac_paths) == 1
    assert aac_paths[0].endswith(".m4a")
    assert encode_calls == [(str(tmp_path / "clip_00001.mp4"), aac_paths[0])]
    assert not Path(aac_paths[0]).exists()  # temp file cleaned up


def test_save_video_raises_for_mp4_when_pyav_is_missing(monkeypatch, tmp_path):
    nodes = _load_nodes()
    _stub_save_env(nodes, monkeypatch, tmp_path)

    seen_paths = []

    def missing_pyav(audio, path):
        seen_paths.append(path)
        raise ImportError("No module named 'av'")

    monkeypatch.setattr(nodes, "_write_waveform_aac", missing_pyav)

    with pytest.raises(ValueError, match="PyAV is not installed"):
        nodes.NeluxSaveVideo().save(
            _fake_component_video(), "out/clip", "auto", "auto", "mp4", "auto", 20, True
        )

    assert len(seen_paths) == 1
    assert not Path(seen_paths[0]).exists()  # temp file cleaned up even on error


def test_save_video_keeps_wav_passthrough_for_mkv(monkeypatch, tmp_path):
    nodes = _load_nodes()
    _stub_save_env(nodes, monkeypatch, tmp_path)

    wav_paths = []
    encode_calls = []

    def fake_write_wav(audio, path):
        wav_paths.append(path)
        Path(path).write_bytes(b"fake-wav")
        return True

    def fake_encode(images, fps, out_path, **kwargs):
        encode_calls.append((out_path, kwargs["passthrough_source"]))
        return out_path

    monkeypatch.setattr(nodes, "_write_waveform_wav", fake_write_wav)
    monkeypatch.setattr(nodes, "_write_waveform_aac", lambda *a, **kw: pytest.fail("should use WAV for mkv"))
    monkeypatch.setattr(nodes, "_nelux_encode", fake_encode)

    nodes.NeluxSaveVideo().save(
        _fake_component_video(), "out/clip", "auto", "auto", "mkv", "auto", 20, True
    )

    assert len(wav_paths) == 1
    assert wav_paths[0].endswith(".wav")
    assert encode_calls == [(str(tmp_path / "clip_00001.mkv"), wav_paths[0])]
    assert not Path(wav_paths[0]).exists()


@pytest.mark.skipif(importlib.util.find_spec("av") is None, reason="PyAV not installed")
@pytest.mark.parametrize("channels,sample_rate", [(2, 44100), (1, 48000), (2, 36000)])
def test_write_waveform_aac_roundtrip(tmp_path, channels, sample_rate):
    import av

    nodes = _load_nodes()
    t = torch.linspace(-0.5, 0.5, steps=sample_rate // 4)
    waveform = torch.stack([t * (1 if c % 2 == 0 else -1) for c in range(channels)]).unsqueeze(0)
    m4a_path = tmp_path / f"out_{channels}_{sample_rate}.m4a"

    assert nodes._write_waveform_aac({"waveform": waveform, "sample_rate": sample_rate}, str(m4a_path)) is True

    with av.open(str(m4a_path)) as container:
        assert len(container.streams.audio) == 1
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "aac"
        assert stream.channels == channels
        decoded_samples = sum(frame.samples for frame in container.decode(audio=0))
        assert decoded_samples > 0
