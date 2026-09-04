import torch

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer
    from flashinfer.activation import gelu_tanh_and_mul, silu_and_mul
    from flashinfer.norm import (fused_add_rmsnorm, fused_add_rmsnorm_quant,
                                 gemma_fused_add_rmsnorm, gemma_rmsnorm,
                                 rmsnorm)
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    @torch.library.custom_op(
        "trtllm::flashinfer_single_prefill_with_kv_cache",
        mutates_args=(),
        device_types="cuda",
    )
    def flashinfer_single_prefill_with_kv_cache(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        o_dtype: torch.dtype,
        custom_mask: torch.Tensor | None,
        sm_scale: float,
        causal: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose FlashInfer single prefill as one torch.compile graph node."""
        if not return_lse:
            raise ValueError(
                "flashinfer_single_prefill_with_kv_cache requires return_lse=True."
            )
        return flashinfer.single_prefill_with_kv_cache(
            query,
            key,
            value,
            o_dtype=o_dtype,
            custom_mask=custom_mask,
            causal=causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
        )

    @flashinfer_single_prefill_with_kv_cache.register_fake
    def _(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        o_dtype: torch.dtype,
        custom_mask: torch.Tensor | None,
        sm_scale: float,
        causal: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not return_lse:
            raise ValueError(
                "flashinfer_single_prefill_with_kv_cache requires return_lse=True."
            )
        del key, value, custom_mask, sm_scale, causal
        lse = torch.empty(
            (query.shape[0], query.shape[1]),
            dtype=torch.float32,
            device=query.device,
        )
        return torch.empty_like(query, dtype=o_dtype), lse

    @torch.library.custom_op(
        "trtllm::flashinfer_nvfp4_attention_sm120",
        mutates_args=(),
        device_types="cuda",
    )
    def flashinfer_nvfp4_attention_sm120(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sm_scale: float,
        causal: bool,
        per_block_mean: bool,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose SM120 NVFP4 QKV quantization and attention as one graph node."""
        quantized_qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
            query,
            key,
            value,
            per_block_mean=per_block_mean,
        )
        return flashinfer.nvfp4_attention_sm120_fwd(
            *quantized_qkv,
            sm_scale=sm_scale,
            causal=causal,
            per_block_mean=per_block_mean,
            out_dtype=out_dtype,
        )

    @flashinfer_nvfp4_attention_sm120.register_fake
    def _(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sm_scale: float,
        causal: bool,
        per_block_mean: bool,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key, value, sm_scale, causal, per_block_mean
        padded_length = ((query.shape[2] + 127) // 128) * 128
        output = query.new_empty(
            (query.shape[0], query.shape[1], padded_length, query.shape[3]),
            dtype=out_dtype,
        )
        lse = torch.empty(
            (query.shape[0], query.shape[1], padded_length),
            dtype=torch.float32,
            device=query.device,
        )
        return output, lse

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    @torch.library.custom_op("trtllm::flashinfer_gelu_tanh_and_mul",
                             mutates_args=())
    def flashinfer_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
        return gelu_tanh_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_gelu_tanh_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps, enable_pdl=get_env_enable_pdl())

    @flashinfer_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_gemma_rmsnorm",
                             mutates_args=())
    def flashinfer_gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                                 eps: float) -> torch.Tensor:
        return gemma_rmsnorm(input,
                             weight,
                             eps,
                             enable_pdl=get_env_enable_pdl())

    @flashinfer_gemma_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        fused_add_rmsnorm(input,
                          residual,
                          weight,
                          eps,
                          enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm_quant",
                             mutates_args=("out", "residual"))
    def flashinfer_fused_add_rmsnorm_quant(out: torch.Tensor,
                                           input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           scale: torch.Tensor,
                                           eps: float) -> None:
        fused_add_rmsnorm_quant(out,
                                input,
                                residual,
                                weight,
                                scale,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    @flashinfer_fused_add_rmsnorm_quant.register_fake
    def _(out: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
          weight: torch.Tensor, scale: torch.Tensor, eps: float) -> None:
        pass

    @torch.library.custom_op("trtllm::flashinfer_gemma_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm(input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float) -> None:
        gemma_fused_add_rmsnorm(input,
                                residual,
                                weight,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op(
        "trtllm::flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=("query", "key"))
    def flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

    @flashinfer_apply_rope_with_cos_sin_cache_inplace.register_fake
    def _(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ):
        return
