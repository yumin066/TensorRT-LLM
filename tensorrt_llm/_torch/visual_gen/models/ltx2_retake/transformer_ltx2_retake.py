# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

"""Retake-specific extensions to the native LTX-2 transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode

from ..ltx2.ltx2_core.adaln import AdaLayerNormSingle
from ..ltx2.ltx2_core.rope import LTXRopeType, apply_rotary_emb
from ..ltx2.transformer_ltx2 import BasicAVTransformerBlock as LTX2BasicAVTransformerBlock
from ..ltx2.transformer_ltx2 import LTX2Attention as LTX2AttentionBase
from ..ltx2.transformer_ltx2 import LTXModel as LTX2ModelBase
from ..ltx2.transformer_ltx2 import LTXModelType
from ..ltx2.transformer_ltx2 import TransformerConfig as LTX2TransformerConfig
from .ltx2_retake_core.attention import FlashInferSelfAttention
from .ltx2_retake_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


class LTX2Attention(LTX2AttentionBase):
    """LTX-2 video self-attention with an optional retake FlashInfer backend."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
        enable_sequence_parallel: bool = False,
        async_ulysses: bool = False,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
            config=config,
            layer_idx=layer_idx,
            module_name=module_name,
            enable_sequence_parallel=enable_sequence_parallel,
            async_ulysses=async_ulysses,
        )
        self.retake_attention: FlashInferSelfAttention | None = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_projected_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.retake_attention is None:
            return super().forward(
                x,
                context=context,
                pe=pe,
                pre_projected_kv=pre_projected_kv,
                key_padding_mask=key_padding_mask,
                timestep=timestep,
            )
        if self.qkv_mode != QKVMode.FUSE_QKV or context is not None or pre_projected_kv is not None:
            raise ValueError("The retake attention backend only supports video self-attention")

        query, key, value = self.get_qkv(x)
        batch_size, sequence_length, _ = query.shape
        if self.qk_norm:
            query = self.norm_q(query)
            key = self.norm_k(key)
        if pe is not None:
            cos, sin = pe
            if cos.ndim == 2:
                cos = cos.view(sequence_length, self.num_attention_heads, self.head_dim).unsqueeze(
                    0
                )
                sin = sin.view(sequence_length, self.num_attention_heads, self.head_dim).unsqueeze(
                    0
                )
            query = apply_rotary_emb(
                query.view(
                    batch_size,
                    sequence_length,
                    self.num_attention_heads,
                    self.head_dim,
                ),
                (cos, sin),
                self.rope_type,
            ).view(batch_size, sequence_length, -1)
            key = apply_rotary_emb(
                key.view(
                    batch_size,
                    sequence_length,
                    self.num_key_value_heads,
                    self.head_dim,
                ),
                (cos, sin),
                self.rope_type,
            ).view(batch_size, sequence_length, -1)

        output = self.retake_attention(
            query,
            key,
            value,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )
        if self.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            output = output.view(
                batch_size,
                sequence_length,
                self.num_attention_heads,
                self.head_dim,
            )
            output = (output * gates.unsqueeze(-1)).view(batch_size, sequence_length, -1)
        return self.to_out[0](output)


@dataclass
class TransformerConfig(LTX2TransformerConfig):
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(LTX2BasicAVTransformerBlock):
    """LTX-2 block extended with TalkVid text cross-attention AdaLN."""

    def _init_video_modules(
        self,
        cfg: TransformerConfig,
        rope_type: LTXRopeType,
        eps: float,
        model_config: Optional["DiffusionModelConfig"],
        idx: int,
    ) -> None:
        _async_ulysses = model_config.parallel.async_ulysses if model_config is not None else False
        self._async_ulysses = _async_ulysses  # block-level gate for v2a async cross-attn
        self.attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            module_name=f"transformer_blocks.{idx}.attn1",
            enable_sequence_parallel=True,
            async_ulysses=_async_ulysses,
        )
        self.attn2 = LTX2AttentionBase(
            query_dim=cfg.dim,
            context_dim=cfg.context_dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            module_name=f"transformer_blocks.{idx}.attn2",
            enable_sequence_parallel=False,
        )
        self.ff = self._make_mlp(cfg, model_config, idx)
        num_adaln_slots = 9 if cfg.cross_attention_adaln else 6
        self.scale_shift_table = nn.Parameter(torch.empty(num_adaln_slots, cfg.dim))
        if cfg.cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, cfg.dim))

    def _init_audio_modules(
        self,
        cfg: TransformerConfig,
        rope_type: LTXRopeType,
        eps: float,
        model_config: "DiffusionModelConfig",
        idx: int,
    ) -> None:
        super()._init_audio_modules(cfg, rope_type, eps, model_config, idx)
        if cfg.cross_attention_adaln:
            self.audio_scale_shift_table = nn.Parameter(torch.empty(9, cfg.dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, cfg.dim))

    # -- AdaLN helpers -------------------------------------------------------

    def _run_text_cross_adaln(
        self,
        query: torch.Tensor,
        args: TransformerArgs,
        attention: LTX2AttentionBase,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        """Apply TalkVid query/KV modulation and gate text cross-attention."""
        if args.prompt_timestep is None:
            raise ValueError("TalkVid cross-attention AdaLN requires prompt_timestep")
        shift_pair, scale_pair, gate_pair = self._get_ada_table_ts_pairs(
            scale_shift_table, query.shape[0], args.timesteps, slice(6, 9)
        )
        query_dtype = query.dtype
        shift_q = (shift_pair[0] + shift_pair[1]).to(query_dtype)
        scale_q = (scale_pair[0] + scale_pair[1]).to(query_dtype)
        gate = (gate_pair[0] + gate_pair[1]).to(query_dtype)
        query = query * (1 + scale_q) + shift_q

        batch_size = query.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(dtype=query_dtype)
            + args.prompt_timestep.reshape(batch_size, args.prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        context_dtype = args.context.dtype
        context = args.context * (1 + scale_kv.to(context_dtype)) + shift_kv.to(context_dtype)
        text_kv = attention.project_kv(context, pe=None)
        return (
            attention(
                query,
                context=context,
                pre_projected_kv=text_kv,
                timestep=args.timesteps,
            )
            * gate
        )

    def _text_cross_attention_input_scale(
        self,
        attention: LTX2AttentionBase,
        args: TransformerArgs,
    ) -> torch.Tensor | None:
        if args.prompt_timestep is not None:
            return None
        return super()._text_cross_attention_input_scale(attention, args)

    def _run_text_cross_attention(
        self,
        query: torch.Tensor,
        args: TransformerArgs,
        attention: LTX2AttentionBase,
        text_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if args.prompt_timestep is None:
            return super()._run_text_cross_attention(query, args, attention, text_kv)
        if attention is self.attn2:
            scale_shift_table = self.scale_shift_table
            prompt_scale_shift_table = self.prompt_scale_shift_table
        elif attention is self.audio_attn2:
            scale_shift_table = self.audio_scale_shift_table
            prompt_scale_shift_table = self.audio_prompt_scale_shift_table
        else:
            raise ValueError("TalkVid AdaLN is only supported for text cross-attention")
        return self._run_text_cross_adaln(
            query,
            args,
            attention,
            scale_shift_table,
            prompt_scale_shift_table,
        )


class LTXModel(LTX2ModelBase):
    """Native LTX-2 transformer with the retake checkpoint extensions."""

    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
        model_config: Optional["DiffusionModelConfig"] = None,
    ) -> None:
        self.cross_attention_adaln = cross_attention_adaln
        super().__init__(
            model_type=model_type,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            timestep_scale_multiplier=timestep_scale_multiplier,
            use_middle_indices_grid=use_middle_indices_grid,
            audio_num_attention_heads=audio_num_attention_heads,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_in_channels=audio_in_channels,
            audio_out_channels=audio_out_channels,
            audio_cross_attention_dim=audio_cross_attention_dim,
            audio_positional_embedding_max_pos=audio_positional_embedding_max_pos,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=apply_gated_attention,
            model_config=model_config,
        )

    # -- Initialization helpers ----------------------------------------------

    def _init_video(
        self, in_channels: int, out_channels: int, caption_channels: int, norm_eps: float
    ) -> None:
        super()._init_video(in_channels, out_channels, caption_channels, norm_eps)
        self.prompt_adaln_single = (
            AdaLayerNormSingle(
                self.inner_dim,
                embedding_coefficient=2,
                make_linear=self._make_linear,
            )
            if self.cross_attention_adaln
            else None
        )
        if self.cross_attention_adaln:
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim,
                embedding_coefficient=9,
                make_linear=self._make_linear,
            )

    def _init_audio(
        self, in_channels: int, out_channels: int, caption_channels: int, norm_eps: float
    ) -> None:
        super()._init_audio(in_channels, out_channels, caption_channels, norm_eps)
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=2,
                make_linear=self._make_linear,
            )
            if self.cross_attention_adaln
            else None
        )
        if self.cross_attention_adaln:
            self.audio_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=9,
                make_linear=self._make_linear,
            )

    def _init_preprocessors(self, cross_pe_max_pos: int | None) -> None:
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                prompt_adaln=self.prompt_adaln_single,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                prompt_adaln=self.audio_prompt_adaln_single,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _prepare_text_kv_cache(
        self, context: torch.Tensor, *, audio: bool
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if self.cross_attention_adaln:
            return None
        return super()._prepare_text_kv_cache(context, audio=audio)

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        apply_gated_attention: bool,
    ) -> None:
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    config=self.model_config,
                )
                for idx in range(num_layers)
            ]
        )
