# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen-Image-Layered transformer variants."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader

from ..qwen_image.transformer_qwen_image import (
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenJointAttention,
    QwenTimestepProjEmbeddings,
    _remap_checkpoint_keys,
    apply_rotary_emb_qwen,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner

_DYNAMIC_QUANT_PARAM_NAMES = (
    "input_scale",
    "weight_scale",
    "weight_scale_2",
)


class QwenEmbedLayer3DRope(QwenEmbedRope):
    """Layer-aware 3D RoPE used by Qwen-Image-Layered.

    The layered checkpoint represents generated RGBA layers followed by
    one conditioning image in ``img_shapes``. Generated layers use their
    layer index as the frame-axis RoPE offset; the conditioning image uses
    the negative frame index from diffusers' reference implementation.
    """

    def forward(
        self,
        video_fhw,
        max_txt_seq_len: int | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        condition_index = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx == condition_index:
                video_freq = self._compute_condition_freqs(frame, height, width, device)
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, condition_index)
        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs = self._pos_freqs_for_device(device)[
            max_vid_index : max_vid_index + max_txt_seq_len_int, ...
        ]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_condition_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs = self._pos_freqs_for_device(device)
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = (
                freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenImageLayeredJointAttention(QwenJointAttention):
    """Qwen-Image-Layered joint attention with masked SageAttention support."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
    ):
        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        config = config or DiffusionModelConfig()
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            dtype=dtype,
            config=config,
            layer_idx=layer_idx,
            module_name=module_name,
        )

        attention_config = config.attention
        if (
            attention_config.backend == "TRTLLM"
            and attention_config.quant_attention_config is not None
        ):
            self.attn_backend = "TRTLLM"
            self.attn = create_attention(
                backend=self.attn_backend,
                layer_idx=self.layer_idx,
                num_heads=self.local_num_attention_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.local_num_key_value_heads,
                quant_config=self.quant_config,
                dtype=self.dtype,
                attention_config=attention_config,
                attention_metadata_state=getattr(config, "attention_metadata_state", None),
                sparse_params=None,
            )

    def _trtllm_masked_attention(
        self,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        txt_v: torch.Tensor,
        img_q: torch.Tensor,
        img_k: torch.Tensor,
        img_v: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_txt = txt_q.shape[1]
        seq_img = img_q.shape[1]
        mask_bool = attention_mask.to(torch.bool)
        expected_mask_len = seq_txt + seq_img
        if mask_bool.shape[1] != expected_mask_len:
            raise ValueError(
                "QwenImageLayeredJointAttention attention_mask length mismatch: "
                f"expected {expected_mask_len}, got {mask_bool.shape[1]}."
            )
        if not torch.all(mask_bool[:, seq_txt:]):
            raise ValueError(
                "QwenImageLayeredJointAttention requires all image tokens to be unmasked."
            )

        outputs = []
        for batch_idx in range(mask_bool.shape[0]):
            txt_indices = torch.nonzero(mask_bool[batch_idx, :seq_txt], as_tuple=False).flatten()
            batch_slice = slice(batch_idx, batch_idx + 1)
            compact_q = torch.cat([txt_q[batch_slice, txt_indices], img_q[batch_slice]], dim=1)
            compact_k = torch.cat([txt_k[batch_slice, txt_indices], img_k[batch_slice]], dim=1)
            compact_v = torch.cat([txt_v[batch_slice, txt_indices], img_v[batch_slice]], dim=1)
            compact_out = self._attn_impl(
                compact_q.flatten(2),
                compact_k.flatten(2),
                compact_v.flatten(2),
                timestep=timestep,
            )

            valid_txt = txt_indices.numel()
            output = torch.zeros(
                (1, expected_mask_len, compact_out.shape[-1]),
                device=compact_out.device,
                dtype=compact_out.dtype,
            )
            output[:, txt_indices] = compact_out[:, :valid_txt]
            output[:, seq_txt:] = compact_out[:, valid_txt:]
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_txt = encoder_hidden_states.shape[1]

        img_q, img_k, img_v = self.get_qkv(hidden_states)
        txt_q = self.add_q_proj(encoder_hidden_states)
        txt_k = self.add_k_proj(encoder_hidden_states)
        txt_v = self.add_v_proj(encoder_hidden_states)

        img_q = img_q.unflatten(-1, (self.heads, -1))
        img_k = img_k.unflatten(-1, (self.heads, -1))
        img_v = img_v.unflatten(-1, (self.heads, -1))
        txt_q = txt_q.unflatten(-1, (self.heads, -1))
        txt_k = txt_k.unflatten(-1, (self.heads, -1))
        txt_v = txt_v.unflatten(-1, (self.heads, -1))

        img_q = self._apply_rms_norm(img_q, self.norm_q)
        img_k = self._apply_rms_norm(img_k, self.norm_k)
        txt_q = self._apply_rms_norm(txt_q, self.norm_added_q)
        txt_k = self._apply_rms_norm(txt_k, self.norm_added_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)

        joint_q = joint_q.transpose(1, 2)
        joint_k = joint_k.transpose(1, 2)
        joint_v = joint_v.transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]

        if attn_mask is None:
            out = self._attn_impl(
                joint_q.transpose(1, 2).flatten(2),
                joint_k.transpose(1, 2).flatten(2),
                joint_v.transpose(1, 2).flatten(2),
                timestep=timestep,
            )
        elif self.attn_backend == "TRTLLM":
            out = self._trtllm_masked_attention(
                txt_q,
                txt_k,
                txt_v,
                img_q,
                img_k,
                img_v,
                attention_mask,
                timestep=timestep,
            )
        else:
            out = F.scaled_dot_product_attention(
                joint_q, joint_k, joint_v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).flatten(2, 3).to(joint_q.dtype)

        txt_attn_output = out[:, :seq_txt, :]
        img_attn_output = out[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output.contiguous())
        txt_attn_output = self.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


class QwenImageLayeredTransformer2DModel(QwenImageTransformer2DModel):
    """Qwen-Image transformer variant for RGBA layer decomposition."""

    def __init__(
        self,
        model_config: Optional["DiffusionModelConfig"] = None,
        *,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        attn_backend: str = "sdpa",
    ):
        super().__init__(
            model_config=model_config,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            axes_dims_rope=axes_dims_rope,
            attn_backend=attn_backend,
        )
        for layer_idx, block in enumerate(self.transformer_blocks):
            block.attn = QwenImageLayeredJointAttention(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=self.model_config.torch_dtype,
                config=self.model_config,
                layer_idx=layer_idx,
                module_name=f"transformer_blocks.{layer_idx}.attn",
            )
        self.apply_quant_config_exclude_modules()

        if use_layer3d_rope:
            self.pos_embed = QwenEmbedLayer3DRope(
                theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
            )
        if use_additional_t_cond:
            self.time_text_embed = QwenTimestepProjEmbeddings(
                embedding_dim=self.inner_dim,
                use_additional_t_cond=True,
            )

    @staticmethod
    def _normalize_img_shapes_for_cuda_graph(*args, **kwargs) -> Optional[Tuple]:
        img_shapes = kwargs.get("img_shapes")
        if img_shapes is None and len(args) > 4:
            img_shapes = args[4]
        if img_shapes is None:
            return None

        def normalize(value):
            if isinstance(value, (list, tuple)):
                return tuple(normalize(item) for item in value)
            return int(value)

        return normalize(img_shapes)

    def register_cuda_graph_extra_key_fns(self, runner: "CUDAGraphRunner") -> None:
        super().register_cuda_graph_extra_key_fns(runner)
        runner.register_extra_key_fn("img_shapes", self._normalize_img_shapes_for_cuda_graph)

    def _dynamic_quant_parameter_names(
        self,
        loader: DynamicLinearWeightLoader,
        weights: Dict[str, torch.Tensor],
    ) -> set[str]:
        """Return scale parameters generated while dynamically quantizing weights."""
        dynamic_quant_params = set()
        for module_name, module in self.named_modules():
            if not isinstance(module, Linear) or module.quant_config is None:
                continue
            quant_algo = loader._get_quant_algo_for_layer(module_name)
            weight_dicts = loader.get_linear_weights(module, module_name, weights)
            if not any(
                loader._should_dynamic_quantize(weight_dict, quant_algo, module_name)
                for weight_dict in weight_dicts
            ):
                continue
            prefix = f"{module_name}." if module_name else ""
            dynamic_quant_params.update(
                f"{prefix}{param_name}"
                for param_name in _DYNAMIC_QUANT_PARAM_NAMES
                if module._parameters.get(param_name) is not None
            )
        return dynamic_quant_params

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load weights and strictly validate dynamic-quant runtime scales."""
        remapped_weights = _remap_checkpoint_keys(weights)
        super().load_weights(weights)

        expected = {name for name, _ in self.named_parameters()}
        loader = DynamicLinearWeightLoader(self.model_config)
        missing = sorted(
            (expected - set(remapped_weights))
            - self._non_serialized_quant_parameter_names()
            - self._dynamic_quant_parameter_names(loader, remapped_weights)
        )
        if missing:
            raise RuntimeError(f"Missing keys when loading transformer: {missing[:5]}...")

    @classmethod
    def from_config_dict(
        cls, cfg: Dict[str, Any], **kwargs
    ) -> "QwenImageLayeredTransformer2DModel":
        """Build from a transformer/config.json dict."""
        return cls(
            patch_size=cfg.get("patch_size", 2),
            in_channels=cfg.get("in_channels", 64),
            out_channels=cfg.get("out_channels", 16),
            num_layers=cfg.get("num_layers", 60),
            attention_head_dim=cfg.get("attention_head_dim", 128),
            num_attention_heads=cfg.get("num_attention_heads", 24),
            joint_attention_dim=cfg.get("joint_attention_dim", 3584),
            axes_dims_rope=tuple(cfg.get("axes_dims_rope", [16, 56, 56])),
            use_additional_t_cond=cfg.get("use_additional_t_cond", False),
            use_layer3d_rope=cfg.get("use_layer3d_rope", False),
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_shapes: Optional[list] = None,
        txt_seq_lens: Optional[list] = None,
        additional_t_cond: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Forward pass with optional Qwen-Image-Layered timestep condition."""
        del kwargs, txt_seq_lens  # Only kept for diffusers API compat.
        missing = []
        if timestep is None:
            missing.append("timestep")
        if img_shapes is None:
            missing.append("img_shapes")
        if missing:
            raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len = encoder_hidden_states.shape[1]
        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)
        image_rotary_emb = self.pos_embed(
            img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
        )

        block_attention_mask = None
        if encoder_hidden_states_mask is not None:
            if encoder_hidden_states_mask.dtype != torch.bool:
                encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones(
                (batch_size, image_seq_len),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            block_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=block_attention_mask,
                timestep=timestep,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)
