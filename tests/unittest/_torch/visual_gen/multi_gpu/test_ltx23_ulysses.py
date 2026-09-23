# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU numerical parity test for LTX-2.3 Ulysses."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, ParallelConfig, TorchCompileConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual_gen_dist_utils import spawn_with_retry  # noqa: E402

_AV_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=64,
    in_channels=16,
    out_channels=16,
    num_layers=2,
    cross_attention_dim=256,
    caption_channels=64,
    norm_eps=1.0e-6,
    positional_embedding_max_pos=[4, 32, 32],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    audio_num_attention_heads=4,
    audio_attention_head_dim=64,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=256,
    audio_positional_embedding_max_pos=[64],
    av_ca_timestep_scale_multiplier=1,
    rope_type=LTXRopeType.SPLIT,
)


def _model_config(ulysses_size: int) -> DiffusionModelConfig:
    if ulysses_size > 1 and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = ulysses_size
        rank = 0
    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        ulysses_size=ulysses_size,
    )
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend="VANILLA"),
        visual_gen_mapping=vgm,
        parallel=ParallelConfig(
            ulysses_size=ulysses_size,
            async_ulysses=False,
        ),
        cache=None,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _init_weights(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and "weight" in name:
                parameter.fill_(1.0)
            elif parameter.numel() > 0:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)


def _positions(*sizes: int, device: torch.device) -> torch.Tensor:
    grids = torch.meshgrid(
        *[torch.arange(size, device=device) for size in sizes],
        indexing="ij",
    )
    indices = torch.stack([grid.flatten() for grid in grids]).float()
    return torch.stack([indices, indices + 1], dim=-1).unsqueeze(0)


def _inputs(device: torch.device, dtype: torch.dtype):
    from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality

    generator = torch.Generator(device=device).manual_seed(456)
    video_positions = _positions(1, 4, 4, device=device)
    audio_positions = _positions(8, device=device)
    video_context = torch.randn(
        1,
        8,
        _AV_CONFIG["cross_attention_dim"],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    audio_context = torch.randn(
        1,
        8,
        _AV_CONFIG["audio_cross_attention_dim"],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    common = dict(
        timesteps=torch.tensor([0.5], device=device),
        sigma=torch.tensor([0.5], device=device),
        cross_modality_sigma=torch.tensor([0.5], device=device),
    )
    video = LTX23Modality(
        latent=torch.randn(
            1,
            16,
            _AV_CONFIG["in_channels"],
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        positions=video_positions,
        context=video_context,
        **common,
    )
    audio = LTX23Modality(
        latent=torch.randn(
            1,
            8,
            _AV_CONFIG["audio_in_channels"],
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        positions=audio_positions,
        context=audio_context,
        **common,
    )
    cache_kwargs = dict(
        video_context=video_context,
        video_positions=video_positions,
        audio_context=audio_context,
        audio_positions=audio_positions,
        dtype=dtype,
    )
    return video, audio, cache_kwargs


def _test_logic(rank: int, world_size: int) -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModelType
    from tensorrt_llm._torch.visual_gen.models.ltx23.transformer_ltx23 import LTX23Model

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    torch.manual_seed(123)
    reference = LTX23Model(
        model_type=LTXModelType.AudioVideo,
        model_config=_model_config(1),
        **_AV_CONFIG,
    ).to(device=device, dtype=dtype)
    reference.eval()
    _init_weights(reference)

    candidate = LTX23Model(
        model_type=LTXModelType.AudioVideo,
        model_config=_model_config(world_size),
        **_AV_CONFIG,
    ).to(device=device, dtype=dtype)
    candidate.eval()
    candidate.load_state_dict(reference.state_dict())

    video, audio, cache_kwargs = _inputs(device, dtype)
    reference_cache = reference.prepare_text_cache(**cache_kwargs)
    candidate_cache = candidate.prepare_text_cache(**cache_kwargs)

    with torch.no_grad():
        reference_video, reference_audio = reference(
            video=video,
            audio=audio,
            text_cache=reference_cache,
        )
        candidate_video, candidate_audio = candidate(
            video=video,
            audio=audio,
            text_cache=candidate_cache,
        )

    torch.testing.assert_close(
        candidate_video,
        reference_video,
        rtol=5.0e-2,
        atol=5.0e-2,
    )
    torch.testing.assert_close(
        candidate_audio,
        reference_audio,
        rtol=5.0e-2,
        atol=5.0e-2,
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        _test_logic(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_ltx23_ulysses_matches_single_gpu(world_size: int) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    spawn_with_retry(
        lambda port: mp.spawn(
            _worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )
    )
