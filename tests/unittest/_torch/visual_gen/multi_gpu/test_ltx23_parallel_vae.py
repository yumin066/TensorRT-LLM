# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU parity test for LTX-2.3 tile-parallel VAE encode."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.video_vae_ltx23 import (
    LTX23VideoEncoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx23.parallel_vae import tile_parallel_encode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual_gen_dist_utils import spawn_with_retry  # noqa: E402

_SMALL_ENCODER_CONFIG = {
    "vae": {
        "dims": 3,
        "in_channels": 3,
        "latent_channels": 4,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "causal_encoder": True,
        "encoder_blocks": [
            ["res_x", {"num_layers": 1}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 1}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 1}],
            ["compress_all", {"multiplier": 1}],
            ["res_x", {"num_layers": 1}],
        ],
    }
}
_VIDEO_SHAPE = (1, 3, 49, 160, 160)
_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(
        tile_size_in_pixels=96,
        tile_overlap_in_pixels=64,
    ),
    temporal_config=TemporalTilingConfig(
        tile_size_in_frames=24,
        tile_overlap_in_frames=16,
    ),
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _init_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _distributed_worker(rank: int, world_size: int, test_fn: Callable, port: int) -> None:
    try:
        _init_worker(rank, world_size, port)
        test_fn(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, test_fn: Callable) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, test_fn, port),
            nprocs=world_size,
            join=True,
        )
    )


def _broadcast_params(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _logic_tile_parallel_encode_parity(rank: int, world_size: int) -> None:
    device = f"cuda:{rank}"
    torch.manual_seed(0)
    encoder = (
        LTX23VideoEncoderConfigurator.from_config(_SMALL_ENCODER_CONFIG)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    encoder.per_channel_statistics.normalize = lambda value: value
    _broadcast_params(encoder)
    video = torch.randn(_VIDEO_SHAPE, device=device, dtype=torch.bfloat16)
    dist.broadcast(video, src=0)
    pg = dist.new_group(list(range(world_size)), use_local_synchronization=False)

    with torch.inference_mode():
        reference = encoder.tiled_encode(video, _TILING_CONFIG)
        actual = tile_parallel_encode(encoder, video, _TILING_CONFIG, pg)

    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)


class TestLTX23ParallelVAEEncode:
    def test_tile_parallel_encode_parity_2gpu(self) -> None:
        _run(2, _logic_tile_parallel_encode_parity)
