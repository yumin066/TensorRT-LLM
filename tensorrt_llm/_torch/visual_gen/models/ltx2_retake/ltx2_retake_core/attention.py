# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer SM120 self-attention kernels used by LTX-2 retake."""

from typing import Literal

import torch

RetakeAttentionBackend = Literal["flashinfer_nvfp4", "flashinfer_fp8"]


class FlashInferSelfAttention:
    """Run a quantized FlashInfer kernel on video self-attention tensors."""

    def __init__(self, backend: RetakeAttentionBackend) -> None:
        self.backend = backend
        if backend == "flashinfer_nvfp4":
            import flashinfer

            required = (
                "nvfp4_attention_sm120_quantize_qkv",
                "nvfp4_attention_sm120_fwd",
            )
            missing = [name for name in required if not hasattr(flashinfer, name)]
            if missing:
                raise ImportError(
                    "FlashInfer NVFP4 attention is unavailable; missing APIs: " + ", ".join(missing)
                )
            self._nvfp4_quantize = flashinfer.nvfp4_attention_sm120_quantize_qkv
            self._nvfp4_forward = flashinfer.nvfp4_attention_sm120_fwd
            self._fp8_forward = None
        elif backend == "flashinfer_fp8":
            try:
                from flashinfer.prefill import fmha_v2_prefill_sm120
            except ImportError as exc:
                raise ImportError(
                    "FlashInfer FP8 attention requires `fmha_v2_prefill_sm120` "
                    "from a compatible FlashInfer build."
                ) from exc
            self._nvfp4_quantize = None
            self._nvfp4_forward = None
            self._fp8_forward = fmha_v2_prefill_sm120
        else:
            raise ValueError(f"Unsupported LTX-2 retake attention backend: {backend!r}")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply the configured kernel to ``[B, T, H * D]`` Q/K/V tensors."""
        if query.device.type != "cuda":
            raise RuntimeError(f"{self.backend} requires a CUDA tensor")
        capability = torch.cuda.get_device_capability(query.device)
        if self.backend == "flashinfer_nvfp4" and capability[0] != 12:
            raise RuntimeError(
                f"{self.backend} requires an SM120-class GPU; got SM{capability[0]}{capability[1]}"
            )
        if self.backend == "flashinfer_fp8" and capability != (12, 0):
            raise RuntimeError(
                f"{self.backend} requires an SM120 GPU; got SM{capability[0]}{capability[1]}"
            )
        if self.backend == "flashinfer_nvfp4":
            if head_dim not in (64, 128):
                raise ValueError(f"FlashInfer NVFP4 attention does not support head_dim={head_dim}")
            return self._run_nvfp4(query, key, value, num_heads, head_dim)
        if head_dim not in (64, 128):
            raise ValueError(f"FlashInfer FP8 attention does not support head_dim={head_dim}")
        return self._run_fp8(query, key, value, num_heads, head_dim)

    @staticmethod
    def _reshape(
        tensor: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, num_heads, head_dim).contiguous()

    @classmethod
    def _reshape_and_pad(
        cls,
        tensor: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, int]:
        tensor = cls._reshape(tensor, num_heads, head_dim)
        sequence_length = tensor.shape[1]
        padded_length = (sequence_length + 127) // 128 * 128
        if padded_length != sequence_length:
            tensor = torch.nn.functional.pad(
                tensor, (0, 0, 0, 0, 0, padded_length - sequence_length)
            )
        return tensor.contiguous(), sequence_length

    def _run_nvfp4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        sequence_length = query.shape[1]
        query = self._reshape(query, num_heads, head_dim)
        key = self._reshape(key, num_heads, head_dim)
        value = self._reshape(value, num_heads, head_dim)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale, correction = self._nvfp4_quantize(
            query, key, value, per_block_mean=False
        )
        output, _ = self._nvfp4_forward(
            q_fp4,
            k_fp4,
            v_fp4,
            q_scale,
            k_scale,
            v_scale,
            correction,
            sm_scale=head_dim**-0.5,
            causal=False,
            per_block_mean=False,
        )
        output = output[:, :, :sequence_length].transpose(1, 2)
        return output.reshape(query.shape[0], sequence_length, num_heads * head_dim)

    @staticmethod
    def _quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_dtype = torch.float8_e4m3fn
        fp8_info = torch.finfo(fp8_dtype)
        scale = (tensor.abs().amax().float() / fp8_info.max).clamp(min=1e-12)
        quantized = (tensor / scale).clamp(min=fp8_info.min, max=fp8_info.max)
        return quantized.to(fp8_dtype).contiguous(), scale.reshape(1)

    def _run_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        query, sequence_length = self._reshape_and_pad(query, num_heads, head_dim)
        key, _ = self._reshape_and_pad(key, num_heads, head_dim)
        value, _ = self._reshape_and_pad(value, num_heads, head_dim)
        query_fp8, query_scale = self._quantize_fp8(query)
        key_fp8, key_scale = self._quantize_fp8(key)
        value_fp8, value_scale = self._quantize_fp8(value)
        output = torch.empty_like(query, dtype=torch.bfloat16)
        qk_scale = query_scale * key_scale / (head_dim**0.5)
        self._fp8_forward(
            query_fp8,
            key_fp8,
            value_fp8,
            output,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=query.shape[1],
            scale_softmax=1.0,
            scale_bmm1=1.0,
            scale_bmm2=1.0,
            scale_bmm1_d=qk_scale,
            scale_bmm2_d=value_scale,
            causal=False,
        )
        output = output[:, :sequence_length]
        return output.reshape(query.shape[0], sequence_length, num_heads * head_dim)
