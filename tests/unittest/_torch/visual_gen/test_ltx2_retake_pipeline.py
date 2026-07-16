# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
    LTX2RetakePipeline,
    _composite_retake_window,
    _fuse_lora_into_transformer_weights,
    _init_retake_latents,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.retake_adapter import LTX2RetakeNativeAdapter
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY


def _minimal_retake_config():
    return DiffusionPipelineConfig(
        model_configs={"transformer": DiffusionModelConfig(pretrained_config=SimpleNamespace())},
        extra_attrs={"workflow": "retake"},
    )


class _StubNativeTransformer:
    """Lightweight stand-in for the native ``LTXModel`` in host-side tests.

    ``LTX2RetakePipeline._init_transformer`` now builds a real native
    transformer; stubbing it keeps construction cheap while still exposing the
    hooks the Modality-aware CUDA graph setup touches.
    """

    def register_cuda_graph_extra_key_fns(self, runner):
        pass

    def forward(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _stub_native_transformer(monkeypatch):
    # Keep host-side construction lightweight: retake builds a native LTXModel
    # in _init_transformer, which would otherwise require a full checkpoint
    # config and materialize the transformer.
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake.build_ltx2_transformer",
        lambda pipeline_config: _StubNativeTransformer(),
    )


def test_ltx2_retake_uses_modality_aware_cuda_graph_runner():
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import _LTX2CUDAGraphRunner

    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    pipeline.pipeline_config.cuda_graph.enable = True
    pipeline.pipeline_config.torch_compile.enable = False
    pipeline._cuda_graph_runners = {}

    pipeline._setup_cuda_graphs()

    # Must use the LTX-2 Modality-aware runner, not the base flat-tensor one.
    assert isinstance(pipeline._cuda_graph_runners.get("transformer"), _LTX2CUDAGraphRunner)


def test_ltx2_retake_cuda_graph_composes_with_torch_compile():
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import _LTX2CUDAGraphRunner

    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    pipeline.pipeline_config.cuda_graph.enable = True
    pipeline.pipeline_config.torch_compile.enable = True
    pipeline._cuda_graph_runners = {}

    pipeline._setup_cuda_graphs()

    # The base _setup_cuda_graphs disables cuda graph when torch_compile is on;
    # the LTX-2 path must still register a runner (the two compose).
    assert isinstance(pipeline._cuda_graph_runners.get("transformer"), _LTX2CUDAGraphRunner)


def test_ltx2_workflow_retake_resolves_retake_variant():
    cfg = SimpleNamespace(extra_attrs={"workflow": "retake"}, cache_backend=None)

    assert LTX2Pipeline.resolve_variant(cfg) is LTX2RetakePipeline


def test_ltx2_workflow_rejects_unknown_value():
    cfg = SimpleNamespace(extra_attrs={"workflow": "unknown"}, cache_backend=None)

    with pytest.raises(ValueError, match="Unsupported LTX-2 workflow"):
        LTX2Pipeline.resolve_variant(cfg)


def test_ltx2_retake_declares_required_extra_params():
    pipeline = LTX2RetakePipeline(_minimal_retake_config())

    specs = pipeline.extra_param_specs

    assert specs["retake_video_path"].type == "str"
    assert specs["retake_start_time"].type == "float"
    assert specs["retake_end_time"].type == "float"
    assert specs["retake_regenerate_video"].default is True
    # Native retake is video-only and preserves source audio, so the default
    # request (which omits the flag) must not request audio regeneration.
    assert specs["retake_regenerate_audio"].default is False
    assert pipeline.default_generation_params == {"num_inference_steps": 40}


def test_ltx2_retake_requires_video_path_before_pipeline_call():
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    req = SimpleNamespace(
        prompt="replacement line",
        params=SimpleNamespace(
            extra_params={
                "retake_start_time": 1.0,
                "retake_end_time": 2.0,
                "retake_regenerate_video": True,
                "retake_regenerate_audio": True,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
            },
            seed=1,
            negative_prompt="",
            num_inference_steps=8,
        ),
    )

    with pytest.raises(ValueError, match="retake_video_path"):
        pipeline.infer(req)


def test_ltx2_retake_reports_optional_dependency_load_errors(monkeypatch):
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    real_import = builtins.__import__

    def fail_ltx_import(name, *args, **kwargs):
        if name.startswith("ltx_core"):
            raise OSError("libtorchaudio.so: undefined symbol")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ltx_import)

    with pytest.raises(ImportError, match="ltx-pipelines"):
        pipeline.load_standard_components(
            "/checkpoint",
            torch.device("cpu"),
            text_encoder_path="/gemma",
        )


def test_ltx2_retake_materializes_video_chunks_as_batched_uint8():
    chunk0 = torch.full((1, 2, 3, 3), 0.25, dtype=torch.float32)
    chunk1 = torch.full((2, 2, 3, 3), 0.5, dtype=torch.float32)

    video = LTX2RetakePipeline._materialize_video(iter([chunk0, chunk1]))

    assert video.shape == (1, 3, 2, 3, 3)
    assert video.dtype == torch.uint8
    assert video[0, 0, 0, 0, 0].item() == 63
    assert video[0, 1, 0, 0, 0].item() == 127


def test_ltx2_retake_normalizes_audio_object():
    audio = SimpleNamespace(waveform=torch.ones(2, 4), sampling_rate=48000)

    waveform, sample_rate = LTX2RetakePipeline._normalize_audio(audio)

    assert waveform.shape == (1, 2, 4)
    assert waveform.dtype == torch.float32
    assert sample_rate == 48000


def test_ltx2_retake_prefers_non_meta_language_model_device():
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=torch.nn.Linear(1, 1)),
        device=torch.device("meta"),
    )

    device = LTX2RetakePipeline._resolve_non_meta_model_device(model)

    assert device.type == "cpu"


def test_ltx2_retake_resolves_offload_mode_from_extra_attrs():
    class OffloadMode(Enum):
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    config = _minimal_retake_config()
    config.extra_attrs["retake_offload_mode"] = "cpu"
    pipeline = LTX2RetakePipeline(config)

    assert pipeline._resolve_offload_mode(OffloadMode) is OffloadMode.CPU


def test_ltx2_registry_accepts_retake_offload_mode_config():
    defaults = PIPELINE_REGISTRY["LTX2Pipeline"].defaults

    assert defaults["retake_offload_mode"] == "none"
    assert defaults["retake_prompt_cache_size"] == 16


class _RecordingStage:
    """Records ``run()`` calls; stands in for upstream ``DiffusionStage``."""

    def __init__(self):
        self.run_calls = []

    def run(  # noqa: PLR0913
        self,
        transformer,
        denoiser,
        sigmas,
        noiser,
        width,
        height,
        frames,
        fps,
        *,
        video,
        audio,
        max_batch_size,
    ):
        self.run_calls.append(
            SimpleNamespace(transformer=transformer, video=video, audio=audio, fps=fps)
        )
        return (
            SimpleNamespace(latent=torch.zeros(1, 2, 8)),
            SimpleNamespace(latent=torch.zeros(1, 2, 4)),
        )


class _FakeResidentRetakePipeline:
    """Resident upstream retake pipeline stub exposing only the pre/post seams
    the native path reuses; ``__call__`` (the eager path) must not run."""

    def __init__(self, distilled=True, has_audio=True):
        self.distilled = distilled
        self.stage = _RecordingStage()
        self.called_directly = False
        self._has_audio = has_audio
        self.image_conditioner = lambda fn: fn(object())
        self.audio_conditioner = lambda fn: fn(object())
        self.prompt_encoder = (
            lambda prompts, *, enhance_first_prompt=False, enhance_prompt_seed=42: [
                SimpleNamespace(
                    video_encoding=torch.zeros(1, 1, 4), audio_encoding=torch.zeros(1, 1, 4)
                )
            ]
        )
        self.video_decoder = lambda latent, tiling, generator: iter(
            [torch.zeros(2, 3, 3, 3, dtype=torch.uint8)]
        )
        self.audio_decoder = lambda latent: SimpleNamespace(
            waveform=torch.ones(2, 4), sampling_rate=48000
        )

    def __call__(self, *args, **kwargs):
        self.called_directly = True
        raise AssertionError("eager RetakePipeline.__call__ must not be used by the native path")


def _fake_upstream_symbols():
    return SimpleNamespace(
        GaussianNoiser=lambda *, generator: SimpleNamespace(generator=generator),
        TemporalRegionMask=lambda *, start_time, end_time, fps: SimpleNamespace(
            start_time=start_time, end_time=end_time, fps=fps
        ),
        DISTILLED_SIGMAS=torch.linspace(1.0, 0.0, 9),
        SimpleDenoiser=lambda *, v_context, a_context: SimpleNamespace(v=v_context, a=a_context),
        video_latent_from_file=lambda *,
        video_encoder,
        file_path,
        output_shape,
        dtype,
        device: torch.zeros(1, 2, 8),
        audio_latent_from_file=lambda *, audio_encoder, file_path, output_shape, dtype, device: (
            torch.zeros(1, 2, 4)
        ),
        ModalitySpec=lambda *, context, conditionings, initial_latent, frozen: SimpleNamespace(
            context=context,
            conditionings=conditionings,
            initial_latent=initial_latent,
            frozen=frozen,
        ),
    )


def _native_retake_req(*, regenerate_video=True, regenerate_audio=False):
    return SimpleNamespace(
        prompt="regenerated window content",
        params=SimpleNamespace(
            extra_params={
                "retake_video_path": "/tmp/src.mp4",
                "retake_start_time": 1.0,
                "retake_end_time": 2.0,
                "retake_regenerate_video": regenerate_video,
                "retake_regenerate_audio": regenerate_audio,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
            },
            seed=42,
            negative_prompt="",
            num_inference_steps=8,
        ),
    )


def _prepare_native_pipeline(monkeypatch, *, distilled=True, has_audio=True):
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    fake = _FakeResidentRetakePipeline(distilled=distilled, has_audio=has_audio)
    pipeline._retake_pipeline = fake
    pipeline._tiling_config = object()
    pipeline._get_videostream_metadata = lambda path: SimpleNamespace(
        fps=24.0, width=64, height=64, frames=9
    )
    monkeypatch.setattr(pipeline, "_import_upstream_retake_symbols", _fake_upstream_symbols)
    return pipeline, fake


def test_ltx2_retake_infer_injects_native_adapter_into_stage_run(monkeypatch):
    pipeline, fake = _prepare_native_pipeline(monkeypatch)

    out = pipeline.infer(_native_retake_req())

    # Denoising goes through DiffusionStage.run with the native adapter, and the
    # eager RetakePipeline.__call__ (which builds its own transformer) is unused.
    assert len(fake.stage.run_calls) == 1
    call = fake.stage.run_calls[0]
    assert isinstance(call.transformer, LTX2RetakeNativeAdapter)
    assert fake.called_directly is False
    # Video window regenerated (one region mask, not frozen); source audio frozen.
    assert call.video.frozen is False
    assert len(call.video.conditionings) == 1
    assert call.audio.frozen is True
    assert call.audio.conditionings == []
    # Output shape flows from the source metadata through the resident decoders.
    assert out.video.dtype == torch.uint8
    assert out.frame_rate == 24.0
    assert out.audio_sample_rate == 48000


def test_ltx2_retake_default_request_omitting_audio_flag_reaches_stage_run(monkeypatch):
    # Regression: an ordinary serving request omits retake_regenerate_audio and
    # relies on DiffusionExecutor._merge_defaults() to fill it from the schema
    # default. That default must be False so the native (video-only) path is
    # reached and the source audio is frozen -- not rejected by the fail-fast.
    pipeline, fake = _prepare_native_pipeline(monkeypatch)
    req = SimpleNamespace(
        prompt="regenerated window content",
        params=SimpleNamespace(
            extra_params={
                "retake_video_path": "/tmp/src.mp4",
                "retake_start_time": 1.0,
                "retake_end_time": 2.0,
                "retake_regenerate_video": True,
                "retake_enhance_prompt": False,
                "retake_max_batch_size": 1,
                # retake_regenerate_audio intentionally omitted.
            },
            seed=42,
            negative_prompt="",
            num_inference_steps=None,
        ),
    )

    # Materialize defaults exactly like the serving worker does.
    DiffusionExecutor._merge_defaults(SimpleNamespace(pipeline=pipeline), req)

    assert req.params.extra_params["retake_regenerate_audio"] is False
    pipeline.infer(req)
    assert len(fake.stage.run_calls) == 1
    assert fake.stage.run_calls[0].audio.frozen is True


def test_ltx2_retake_infer_rejects_audio_regeneration(monkeypatch):
    pipeline, fake = _prepare_native_pipeline(monkeypatch)

    with pytest.raises(NotImplementedError, match="regenerate_audio"):
        pipeline.infer(_native_retake_req(regenerate_audio=True))

    assert fake.stage.run_calls == []
    assert fake.called_directly is False


def test_ltx2_retake_infer_rejects_video_preservation(monkeypatch):
    pipeline, fake = _prepare_native_pipeline(monkeypatch)

    with pytest.raises(NotImplementedError, match="regenerate_video"):
        pipeline.infer(_native_retake_req(regenerate_video=False))

    assert fake.stage.run_calls == []


def test_ltx2_retake_infer_requires_distilled_schedule(monkeypatch):
    pipeline, fake = _prepare_native_pipeline(monkeypatch, distilled=False)

    with pytest.raises(NotImplementedError, match="distilled"):
        pipeline.infer(_native_retake_req())

    assert fake.stage.run_calls == []


def _write_lora(path, entries):
    import safetensors.torch

    safetensors.torch.save_file(entries, path)


def test_fuse_lora_applies_scaled_ba_and_strips_comfy_prefix(tmp_path):
    wkey = "transformer_blocks.0.attn.to_q.weight"
    weights = {wkey: torch.zeros(2, 3)}
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.tensor(
                [[1.0, 2.0, 3.0]]
            ),  # (r=1, in=3)
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.tensor(
                [[1.0], [2.0]]
            ),  # (out=2, r=1)
        },
    )

    _fuse_lora_into_transformer_weights(weights, lora_path, strength=0.5)

    # W += 0.5 * (B @ A); B@A = [[1,2,3],[2,4,6]] -> scaled [[.5,1,1.5],[1,2,3]].
    assert torch.allclose(weights[wkey], torch.tensor([[0.5, 1.0, 1.5], [1.0, 2.0, 3.0]]))


def test_fuse_lora_raises_when_no_module_matches(tmp_path):
    weights = {"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 3)}
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "diffusion_model.nonexistent.module.lora_A.weight": torch.zeros(1, 3),
            "diffusion_model.nonexistent.module.lora_B.weight": torch.zeros(2, 1),
        },
    )

    with pytest.raises(ValueError, match="fused 0 modules"):
        _fuse_lora_into_transformer_weights(weights, lora_path, strength=1.0)


def test_load_weights_fuses_configured_lora(tmp_path):
    wkey = "transformer_blocks.0.attn.to_q.weight"
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.tensor(
                [[1.0, 1.0, 1.0]]
            ),
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.tensor(
                [[1.0], [1.0]]
            ),
        },
    )
    cfg = _minimal_retake_config()
    cfg.extra_attrs["retake_lora_path"] = lora_path
    cfg.extra_attrs["retake_lora_strength"] = 2.0
    pipeline = LTX2RetakePipeline(cfg)

    captured = {}

    class _CapturingTransformer:
        def load_weights(self, weights):
            captured["weights"] = weights

    pipeline.transformer = _CapturingTransformer()
    pipeline.load_weights({wkey: torch.zeros(2, 3)})

    # B @ A contracts over rank r=1, so all-ones (2,1)@(1,3) -> ones(2,3);
    # strength 2.0 over a zero base -> every entry 2.0.
    assert torch.allclose(captured["weights"][wkey], torch.full((2, 3), 2.0))


def test_fuse_lora_supports_lora_down_up_suffix(tmp_path):
    wkey = "transformer_blocks.0.attn1.to_q.weight"
    weights = {wkey: torch.zeros(2, 3)}
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_down.weight": torch.tensor(
                [[1.0, 2.0, 3.0]]
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_up.weight": torch.tensor(
                [[1.0], [2.0]]
            ),
        },
    )

    _fuse_lora_into_transformer_weights(weights, lora_path, strength=1.0)

    assert torch.allclose(weights[wkey], torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))


def test_fuse_lora_applies_alpha_scaling(tmp_path):
    wkey = "transformer_blocks.0.attn1.to_q.weight"
    weights = {wkey: torch.zeros(2, 2)}
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.eye(2),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.eye(2),
            "diffusion_model.transformer_blocks.0.attn1.to_q.alpha": torch.tensor([1.0]),
        },
    )

    # rank=2, alpha=1 -> scale = strength * alpha/rank = 1 * 1/2 = 0.5; B@A = I.
    _fuse_lora_into_transformer_weights(weights, lora_path, strength=1.0)

    assert torch.allclose(weights[wkey], 0.5 * torch.eye(2))


def test_fuse_lora_strips_model_diffusion_model_prefix(tmp_path):
    wkey = "transformer_blocks.0.attn1.to_q.weight"
    weights = {wkey: torch.zeros(2, 3)}
    lora_path = str(tmp_path / "lora.safetensors")
    _write_lora(
        lora_path,
        {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1, 3),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.ones(2, 1),
        },
    )

    _fuse_lora_into_transformer_weights(weights, lora_path, strength=1.0)

    assert torch.allclose(weights[wkey], torch.ones(2, 3))


def test_load_weights_without_lora_is_unchanged(tmp_path):
    wkey = "transformer_blocks.0.attn.to_q.weight"
    pipeline = LTX2RetakePipeline(_minimal_retake_config())
    captured = {}

    class _CapturingTransformer:
        def load_weights(self, weights):
            captured["weights"] = weights

    pipeline.transformer = _CapturingTransformer()
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pipeline.load_weights({wkey: base})

    assert torch.equal(captured["weights"][wkey], base)


def test_retake_pixel_window_rounds_and_returns_half_open_span():
    assert _retake_pixel_window(1.0, 2.0, 24.0, 100) == (24, 48)
    # round(1.48*24)=round(35.52)=36, round(2.02*24)=round(48.48)=48
    assert _retake_pixel_window(1.48, 2.02, 24.0, 100) == (36, 48)


def test_retake_pixel_window_clamps_out_of_range_times():
    assert _retake_pixel_window(-1.0, 10.0, 24.0, 100) == (0, 100)
    assert _retake_pixel_window(0.0, 100.0, 24.0, 50) == (0, 50)


def test_retake_pixel_window_inverted_times_yield_empty_span():
    start, end = _retake_pixel_window(2.0, 1.0, 24.0, 100)
    assert start == end == 48  # end clamped up to start -> empty


def test_composite_replaces_window_and_keeps_outside_byte_identical():
    source = torch.arange(5 * 2 * 2 * 3, dtype=torch.uint8).reshape(1, 5, 2, 2, 3)
    window = torch.full((1, 2, 2, 2, 3), 200, dtype=torch.uint8)

    out = _composite_retake_window(source, window, 1, 3)

    assert torch.equal(out[:, 0], source[:, 0])  # before window: byte-identical
    assert torch.equal(out[:, 3:5], source[:, 3:5])  # after window: byte-identical
    assert torch.equal(out[:, 1:3], window)  # window replaced
    assert torch.equal(source, torch.arange(60, dtype=torch.uint8).reshape(1, 5, 2, 2, 3))


def test_composite_empty_window_returns_unmodified_copy():
    source = torch.arange(3 * 2 * 2 * 3, dtype=torch.uint8).reshape(1, 3, 2, 2, 3)
    window = torch.zeros(1, 0, 2, 2, 3, dtype=torch.uint8)

    out = _composite_retake_window(source, window, 2, 2)

    assert torch.equal(out, source)
    assert out.data_ptr() != source.data_ptr()  # a copy, not the same storage


def test_composite_rejects_wrong_window_length():
    source = torch.zeros(1, 5, 2, 2, 3, dtype=torch.uint8)
    window = torch.zeros(1, 3, 2, 2, 3, dtype=torch.uint8)  # span is 2, not 3
    with pytest.raises(ValueError, match="window frame count"):
        _composite_retake_window(source, window, 1, 3)


def test_composite_rejects_shape_mismatch():
    source = torch.zeros(1, 5, 2, 2, 3, dtype=torch.uint8)
    window = torch.zeros(1, 2, 4, 2, 3, dtype=torch.uint8)  # H differs
    with pytest.raises(ValueError, match="shape mismatch"):
        _composite_retake_window(source, window, 1, 3)


def test_conditioned_latent_ranges_internal_window():
    # ratio=8, 97 pixel frames -> L = (97-1)//8+1 = 13 latent frames.
    # pixel [16,40): lat_start=(16-1)//8+1=2; lat_end=((39-1)//8+1)+1=6.
    latent_window, cond = _retake_conditioned_latent_ranges(16, 40, 97, 8)
    assert latent_window == (2, 6)
    assert cond == [(0, 2), (6, 13)]


def test_conditioned_latent_ranges_full_window_has_no_context():
    latent_window, cond = _retake_conditioned_latent_ranges(0, 97, 97, 8)
    assert latent_window == (0, 13)
    assert cond == []  # everything regenerated


def test_conditioned_latent_ranges_empty_window_is_all_context():
    latent_window, cond = _retake_conditioned_latent_ranges(40, 40, 97, 8)
    assert latent_window == (0, 0)
    assert cond == [(0, 13)]  # everything conditioned


def test_conditioned_latent_ranges_leading_and_trailing_only():
    # From frame 0: no leading context range.
    lw, cond = _retake_conditioned_latent_ranges(0, 24, 97, 8)
    assert lw == (0, 4)
    assert cond == [(4, 13)]
    # To the last frame: no trailing context range.
    lw, cond = _retake_conditioned_latent_ranges(73, 97, 97, 8)
    assert lw == (10, 13)
    assert cond == [(0, 10)]


def test_conditioned_latent_ranges_quantizes_subframe_windows():
    # Two pixel windows inside the same latent-frame boundaries collapse to the
    # same latent window (VAE temporal granularity).
    a, _ = _retake_conditioned_latent_ranges(17, 33, 97, 8)
    b, _ = _retake_conditioned_latent_ranges(19, 33, 97, 8)
    assert a == b == (3, 5)


def test_conditioned_latent_ranges_ratio_one_is_frame_identity():
    lw, cond = _retake_conditioned_latent_ranges(2, 5, 10, 1)
    assert lw == (2, 5)
    assert cond == [(0, 2), (5, 10)]


def test_conditioned_latent_ranges_rejects_bad_ratio_or_frames():
    with pytest.raises(ValueError, match="temporal_ratio"):
        _retake_conditioned_latent_ranges(0, 8, 97, 0)
    with pytest.raises(ValueError, match="num_frames"):
        _retake_conditioned_latent_ranges(0, 8, -1, 8)


def test_init_retake_latents_context_from_source_window_from_noise():
    noise = torch.ones(1, 2, 5, 2, 2)
    source = torch.full((1, 2, 5, 2, 2), 2.0)

    out = _init_retake_latents(noise, source, [(0, 1), (4, 5)])

    assert torch.equal(out[:, :, 0], source[:, :, 0])  # leading context from source
    assert torch.equal(out[:, :, 4], source[:, :, 4])  # trailing context from source
    assert torch.equal(out[:, :, 1:4], noise[:, :, 1:4])  # regenerated window stays noise
    assert torch.equal(noise, torch.ones(1, 2, 5, 2, 2))  # input unmodified


def test_init_retake_latents_full_window_keeps_all_noise():
    noise = torch.randn(1, 2, 5, 2, 2)
    source = torch.zeros(1, 2, 5, 2, 2)

    out = _init_retake_latents(noise, source, [])

    assert torch.equal(out, noise)


def test_init_retake_latents_clamps_out_of_range():
    noise = torch.ones(1, 2, 5, 2, 2)
    source = torch.full((1, 2, 5, 2, 2), 2.0)

    out = _init_retake_latents(noise, source, [(-2, 1), (4, 99)])

    assert torch.equal(out[:, :, 0], source[:, :, 0])
    assert torch.equal(out[:, :, 4], source[:, :, 4])
    assert torch.equal(out[:, :, 1:4], noise[:, :, 1:4])


def test_init_retake_latents_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        _init_retake_latents(torch.ones(1, 2, 5, 2, 2), torch.ones(1, 2, 4, 2, 2), [(0, 1)])


def test_init_retake_latents_rejects_non_5d():
    with pytest.raises(ValueError, match=r"B, C, T, H, W"):
        _init_retake_latents(torch.ones(2, 5, 2, 2), torch.ones(2, 5, 2, 2), [(0, 1)])


def test_fuse_lora_accepts_directory_of_shards(tmp_path):
    wkey = "transformer_blocks.0.attn1.to_q.weight"
    weights = {wkey: torch.zeros(2, 3)}
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    _write_lora(
        str(lora_dir / "lora.safetensors"),
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1, 3),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.ones(2, 1),
        },
    )

    _fuse_lora_into_transformer_weights(weights, str(lora_dir), strength=1.0)

    assert torch.allclose(weights[wkey], torch.ones(2, 3))


def test_ltx2_retake_prompt_encoder_cache_reuses_matching_prompts():
    class PromptEncoder:
        def __init__(self):
            self.calls = 0
            self._trtllm_prompt_cache_size = 2

        def __call__(
            self,
            prompts,
            *,
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=42,
        ):
            self.calls += 1
            return [(tuple(prompts), self.calls)]

    LTX2RetakePipeline._install_prompt_encoder_cache(PromptEncoder)
    encoder = PromptEncoder()

    assert encoder(["hello"]) == [(("hello",), 1)]
    assert encoder(["hello"]) == [(("hello",), 1)]
    assert encoder.calls == 1

    assert encoder(["other"]) == [(("other",), 2)]
    assert encoder(["third"]) == [(("third",), 3)]
    assert encoder(["hello"]) == [(("hello",), 4)]
