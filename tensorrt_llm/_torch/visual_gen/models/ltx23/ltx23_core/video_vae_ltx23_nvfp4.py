# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 W1 NVFP4 model variants for the LTX-2.3 Retake video VAE.

Both recipes quantize the two 3x3x3 convolutions in every ResNet block.  The
decoder additionally quantizes its four upsample convolutions.  Encoder and
decoder input/output convolutions and encoder downsample convolutions stay in
BF16.  Conv3d outputs and residual additions also remain BF16.

PixelNorm, SiLU, temporal replication, spatial zero padding, and C16 NVFP4
activation quantization are fused into one CUDA preprocessing kernel for
ResNet convolutions.  Upsample convolutions use the same kernel without
PixelNorm or SiLU.  FP32 global activation multipliers come from representative
BF16 calibration runs; C16 E4M3 scales remain dynamic.  The Retake pipeline
constructs these model variants before checkpoint loading and prepares their
weights once from ``post_load_weights``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ...ltx2.ltx2_core.normalization import PixelNorm
from ...ltx2.ltx2_core.video_vae.convolution import CausalConv3d
from ...ltx2.ltx2_core.video_vae.resnet import ResnetBlock3D
from ...ltx2.ltx2_core.video_vae.sampling import DepthToSpaceUpsample
from .video_vae_ltx23 import (
    LTX23VideoDecoder,
    LTX23VideoDecoderConfigurator,
    LTX23VideoEncoder,
    LTX23VideoEncoderConfigurator,
)

NVFP4_GLOBAL_QUANT_MAX = 448.0 * 6.0
_RETAKE_RECIPE_FILENAME = "video_vae_ltx23_retake_nvfp4_global_fp32_scales.json"


@dataclass(frozen=True)
class LTX23Nvfp4RetakeRecipe:
    """Validated static activation multipliers for one Retake pipeline."""

    encoder: Mapping[str, float]
    decoder: Mapping[str, float]
    metadata: Mapping[str, Any]


def load_ltx23_nvfp4_retake_recipe(
    path: str | Path | None = None,
) -> LTX23Nvfp4RetakeRecipe:
    """Load the pipeline-scoped offline calibration artifact.

    Only the scalar FP32 global multiplier is static.  The fused preprocessing
    kernel still computes one E4M3 scale for every 16 activation values at
    runtime.
    """

    recipe_path = (
        Path(path) if path is not None else Path(__file__).with_name(_RETAKE_RECIPE_FILENAME)
    )
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported LTX-2.3 NVFP4 recipe schema: {recipe_path}")
    if payload.get("model") != "LTX-2.3" or payload.get("pipeline") != "retake":
        raise ValueError(
            f"NVFP4 recipe is not scoped to the LTX-2.3 Retake pipeline: {recipe_path}"
        )

    def read_section(name: str) -> dict[str, float]:
        section = payload.get(name)
        if not isinstance(section, dict) or not section:
            raise ValueError(f"NVFP4 recipe has no non-empty {name!r} section")
        values = {}
        for site, record in section.items():
            if not isinstance(record, dict):
                raise TypeError(f"NVFP4 recipe entry {name}.{site} must be an object")
            value = float(record.get("activation_quant_multiplier", 0.0))
            if value <= 0.0:
                raise ValueError(f"NVFP4 recipe entry {name}.{site} has an invalid multiplier")
            values[site] = value
        return values

    metadata = {key: value for key, value in payload.items() if key not in {"encoder", "decoder"}}
    return LTX23Nvfp4RetakeRecipe(
        encoder=read_section("encoder"),
        decoder=read_section("decoder"),
        metadata=metadata,
    )


@dataclass(frozen=True)
class EncoderResnetSite:
    """One replaceable encoder ResNet block."""

    name: str
    parent: nn.ModuleList
    index: int
    module: ResnetBlock3D


@dataclass(frozen=True)
class DecoderResnetSite:
    """One replaceable decoder ResNet block."""

    name: str
    parent: nn.ModuleList
    index: int
    module: ResnetBlock3D


@dataclass(frozen=True)
class DecoderUpsampleSite:
    """One replaceable decoder upsample Conv3d."""

    name: str
    module: DepthToSpaceUpsample


def encoder_resnet_sites(encoder: LTX23VideoEncoder) -> list[EncoderResnetSite]:
    """Return encoder ResNet sites in execution order."""

    sites: list[EncoderResnetSite] = []
    for down_index, down_block in enumerate(encoder.down_blocks):
        if hasattr(down_block, "res_blocks"):
            for res_index, module in enumerate(down_block.res_blocks):
                if not isinstance(module, ResnetBlock3D):
                    continue
                sites.append(
                    EncoderResnetSite(
                        name=f"down_blocks.{down_index}.res_blocks.{res_index}",
                        parent=down_block.res_blocks,
                        index=res_index,
                        module=module,
                    )
                )
        elif isinstance(down_block, ResnetBlock3D):
            sites.append(
                EncoderResnetSite(
                    name=f"down_blocks.{down_index}",
                    parent=encoder.down_blocks,
                    index=down_index,
                    module=down_block,
                )
            )
    if not sites:
        raise ValueError("LTX-2.3 encoder contains no ResnetBlock3D modules")
    return sites


def decoder_resnet_sites(decoder: LTX23VideoDecoder) -> list[DecoderResnetSite]:
    """Return decoder ResNet sites in execution order."""

    sites: list[DecoderResnetSite] = []
    for up_index, up_block in enumerate(decoder.up_blocks):
        if hasattr(up_block, "res_blocks"):
            for res_index, module in enumerate(up_block.res_blocks):
                if not isinstance(module, ResnetBlock3D):
                    continue
                sites.append(
                    DecoderResnetSite(
                        name=f"up_blocks.{up_index}.res_blocks.{res_index}",
                        parent=up_block.res_blocks,
                        index=res_index,
                        module=module,
                    )
                )
        elif isinstance(up_block, ResnetBlock3D):
            sites.append(
                DecoderResnetSite(
                    name=f"up_blocks.{up_index}",
                    parent=decoder.up_blocks,
                    index=up_index,
                    module=up_block,
                )
            )
    if not sites:
        raise ValueError("LTX-2.3 decoder contains no ResnetBlock3D modules")
    return sites


def decoder_upsample_sites(decoder: LTX23VideoDecoder) -> list[DecoderUpsampleSite]:
    """Return decoder upsample Conv3d sites in execution order."""

    sites = [
        DecoderUpsampleSite(name=f"up_blocks.{index}.conv", module=module)
        for index, module in enumerate(decoder.up_blocks)
        if isinstance(module, DepthToSpaceUpsample)
    ]
    if not sites:
        raise ValueError("LTX-2.3 decoder contains no DepthToSpaceUpsample modules")
    return sites


@cache
def _preprocess_module():
    from flashinfer.jit.core import gen_jit_spec, sm120a_nvcc_flags

    source = Path(__file__).with_name("video_vae_ltx23_nvfp4_preprocess.cu")
    return gen_jit_spec(
        "video_vae_ltx23_nvfp4_preprocess_v1",
        [source],
        extra_cuda_cflags=sm120a_nvcc_flags,
    ).build_and_load()


def _validate_conv(source: CausalConv3d, name: str) -> None:
    if not isinstance(source, CausalConv3d):
        raise TypeError(f"{name} must be CausalConv3d, got {type(source).__name__}")
    conv = source.conv
    expected = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 1, 1),
        "dilation": (1, 1, 1),
        "groups": 1,
        "padding_mode": "zeros",
    }
    observed = {key: getattr(conv, key) for key in expected}
    if observed != expected:
        raise ValueError(f"{name} has unsupported Conv3d attributes: {observed}")
    if source.in_channels % 128 or source.out_channels % 128:
        raise ValueError(f"{name} input/output channels must be multiples of 128")


class _PreparedNvfp4Conv3d(nn.Module):
    def __init__(self, source: CausalConv3d, input_scale: float, name: str) -> None:
        super().__init__()
        _validate_conv(source, name)
        if float(input_scale) <= 0.0:
            raise ValueError(f"{name} requires one positive activation multiplier")
        # Keep the unpacked nn.Conv3d under the same ``conv`` name used by
        # CausalConv3d.  The BF16 checkpoint therefore loads directly into the
        # final graph without any post-load module replacement.
        self.conv: nn.Conv3d | None = source.conv
        self.input_scale = float(input_scale)
        self.name = name
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self._prepared = False

    def post_load_weights(self) -> None:
        """Pack the checkpoint weight exactly once after the model reaches CUDA."""

        if self._prepared:
            return
        if self.conv is None:
            raise RuntimeError(f"{self.name} has no unpacked checkpoint weight")
        device = self.conv.weight.device
        if not device.type == "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
            raise RuntimeError("LTX-2.3 NVFP4 VAE requires an SM120 GPU")
        try:
            from flashinfer import prepare_nvfp4_conv3d_weight
        except ImportError as error:
            raise RuntimeError("LTX-2.3 NVFP4 VAE requires FlashInfer PR #4176") from error

        packed_weight, weight_scale, weight_global_scale = prepare_nvfp4_conv3d_weight(
            self.conv.weight.detach()
        )
        input_global_scale = torch.tensor([self.input_scale], device=device, dtype=torch.float32)
        alpha = torch.reciprocal(input_global_scale * weight_global_scale)
        alpha_and_bias = torch.cat(
            (
                alpha,
                torch.zeros_like(alpha),
                self.conv.bias.detach().to(torch.float32),
            )
        ).contiguous()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("input_global_scale", input_global_scale)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("alpha_and_bias", alpha_and_bias)
        self.conv = None
        self._prepared = True

    def forward(
        self,
        value: torch.Tensor,
        *,
        causal: bool,
        eps: float = 1.0e-8,
        normalize_and_activate: bool = False,
    ) -> torch.Tensor:
        if not self._prepared:
            raise RuntimeError(f"{self.name} must run post_load_weights() before inference")
        from flashinfer.conv.nvfp4_sm120 import run_sm120_nvfp4_conv3d

        batch, channels, depth, height, width = value.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {channels}")
        packed = torch.empty(
            (batch, depth + 2, height + 2, width + 2, channels // 2),
            device=value.device,
            dtype=torch.uint8,
        )
        scales = torch.empty(
            (batch, depth + 2, height + 2, width + 2, channels // 16),
            device=value.device,
            dtype=torch.uint8,
        )
        _preprocess_module().video_vae_ltx23_nvfp4_preprocess(
            value,
            self.input_global_scale,
            packed,
            scales,
            int(causal),
            int(normalize_and_activate),
            float(eps),
        )
        output = torch.empty(
            (batch, depth, height, width, self.out_channels),
            device=value.device,
            dtype=torch.bfloat16,
        )
        run_sm120_nvfp4_conv3d(
            packed,
            self.packed_weight,
            scales,
            self.weight_scale,
            self.alpha_and_bias,
            output,
            fuse_bias=True,
        )
        return output.permute(0, 4, 1, 2, 3)


class LTX23Nvfp4ResnetBlock(nn.Module):
    """Drop-in LTX-2.3 ResNet block with both Conv3d sites in W4A4."""

    def __init__(
        self,
        source: ResnetBlock3D,
        scales: Sequence[float],
        name: str,
    ) -> None:
        super().__init__()
        if len(scales) != 2 or any(float(scale) <= 0.0 for scale in scales):
            raise ValueError(f"{name} requires two positive activation scales")
        if not isinstance(source.norm1, PixelNorm) or not isinstance(source.norm2, PixelNorm):
            raise TypeError(f"{name} requires PixelNorm")
        if source.inject_noise or source.timestep_conditioning:
            raise ValueError(f"{name} cannot use noisy or conditioned blocks")
        if source.dropout.p != 0:
            raise ValueError(f"{name} requires dropout=0")
        self.conv1 = _PreparedNvfp4Conv3d(source.conv1, float(scales[0]), f"{name}.conv1")
        self.conv2 = _PreparedNvfp4Conv3d(source.conv2, float(scales[1]), f"{name}.conv2")
        self.conv_shortcut = source.conv_shortcut
        self.norm3 = source.norm3
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.eps1 = float(source.norm1.eps)
        self.eps2 = float(source.norm2.eps)

    def forward(
        self,
        input_tensor: torch.Tensor,
        causal: bool = True,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if timestep is not None:
            raise ValueError("NVFP4 VAE ResNet block does not support timestep conditioning")
        del generator
        residual = self.conv_shortcut(self.norm3(input_tensor))
        hidden = self.conv1(
            input_tensor,
            causal=causal,
            eps=self.eps1,
            normalize_and_activate=True,
        )
        hidden = self.conv2(
            hidden,
            causal=causal,
            eps=self.eps2,
            normalize_and_activate=True,
        )
        return residual + hidden


def _validated_scales(
    activation_scales: Mapping[str, float],
    expected: set[str],
    recipe: str,
) -> dict[str, float]:
    observed = set(activation_scales)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"activation scale keys do not match stable {recipe} recipe: missing={missing}, "
            f"unexpected={unexpected}"
        )

    validated: dict[str, float] = {}
    for name, value in activation_scales.items():
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} requires a positive activation multiplier")
        validated[name] = value
    return validated


def _build_nvfp4_encoder_graph(
    encoder: LTX23VideoEncoder, activation_scales: Mapping[str, float]
) -> tuple[str, ...]:
    sites = encoder_resnet_sites(encoder)
    expected = {
        conv_name for site in sites for conv_name in (f"{site.name}.conv1", f"{site.name}.conv2")
    }
    scales = _validated_scales(activation_scales, expected, "encoder")
    installed = []
    for site in sites:
        site.parent[site.index] = LTX23Nvfp4ResnetBlock(
            site.module,
            (scales[f"{site.name}.conv1"], scales[f"{site.name}.conv2"]),
            site.name,
        ).eval()
        installed.append(site.name)
    return tuple(installed)


def _build_nvfp4_decoder_graph(
    decoder: LTX23VideoDecoder, activation_scales: Mapping[str, float]
) -> tuple[str, ...]:
    resnet_sites = decoder_resnet_sites(decoder)
    upsample_sites = decoder_upsample_sites(decoder)
    expected = {
        conv_name
        for site in resnet_sites
        for conv_name in (f"{site.name}.conv1", f"{site.name}.conv2")
    }
    expected.update(site.name for site in upsample_sites)
    scales = _validated_scales(activation_scales, expected, "decoder")
    installed = []
    for site in resnet_sites:
        site.parent[site.index] = LTX23Nvfp4ResnetBlock(
            site.module,
            (scales[f"{site.name}.conv1"], scales[f"{site.name}.conv2"]),
            site.name,
        ).eval()
        installed.append(site.name)
    for site in upsample_sites:
        site.module.conv = _PreparedNvfp4Conv3d(
            site.module.conv, scales[site.name], site.name
        ).eval()
        installed.append(site.name)
    return tuple(installed)


class _LTX23Nvfp4VideoVaeMixin:
    """Finalize all pre-constructed W4A4 Conv3d modules after weight load."""

    nvfp4_sites: tuple[str, ...]

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, _PreparedNvfp4Conv3d):
                module.post_load_weights()


class LTX23Nvfp4VideoEncoder(_LTX23Nvfp4VideoVaeMixin, LTX23VideoEncoder):
    """Retake encoder with all 18 ResNet blocks represented by NVFP4 modules."""

    def __init__(
        self,
        *args,
        activation_scales: Mapping[str, float],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nvfp4_sites = _build_nvfp4_encoder_graph(self, activation_scales)


class LTX23Nvfp4VideoDecoder(_LTX23Nvfp4VideoVaeMixin, LTX23VideoDecoder):
    """Retake decoder with 18 ResNet blocks and four upsample Conv3d in NVFP4."""

    def __init__(
        self,
        *args,
        activation_scales: Mapping[str, float],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nvfp4_sites = _build_nvfp4_decoder_graph(self, activation_scales)


class LTX23Nvfp4VideoEncoderConfigurator(LTX23VideoEncoderConfigurator):
    model_cls = LTX23Nvfp4VideoEncoder


class LTX23Nvfp4VideoDecoderConfigurator(LTX23VideoDecoderConfigurator):
    model_cls = LTX23Nvfp4VideoDecoder


__all__ = [
    "LTX23Nvfp4RetakeRecipe",
    "LTX23Nvfp4ResnetBlock",
    "LTX23Nvfp4VideoDecoder",
    "LTX23Nvfp4VideoDecoderConfigurator",
    "LTX23Nvfp4VideoEncoder",
    "LTX23Nvfp4VideoEncoderConfigurator",
    "decoder_resnet_sites",
    "decoder_upsample_sites",
    "encoder_resnet_sites",
    "load_ltx23_nvfp4_retake_recipe",
]
