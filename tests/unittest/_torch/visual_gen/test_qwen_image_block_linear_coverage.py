# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from scripts.visualgen_eval import qwen_image_block_linear_coverage as coverage  # noqa: E402


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.Sequential(nn.Linear(4, 4))
        self.add_q_proj = nn.Linear(4, 4)
        self.add_k_proj = nn.Linear(4, 4)
        self.add_v_proj = nn.Linear(4, 4)
        self.to_add_out = nn.Linear(4, 4)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.Identity(), nn.Linear(4, 4))
        self.txt_mod = nn.Sequential(nn.Identity(), nn.Linear(4, 4))
        self.attn = _Attention()
        self.img_mlp = _Mlp()
        self.txt_mlp = _Mlp()


class _NormOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _Transformer(nn.Module):
    def __init__(self, num_layers: int = coverage.QWEN_IMAGE_LAYER_COUNT) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.img_in = nn.Linear(4, 4)
        self.txt_in = nn.Linear(4, 4)
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(num_layers)])
        self.norm_out = _NormOut()
        self.proj_out = nn.Linear(4, 4)


def test_analyze_transformer_block_linear_coverage_matches_840_contract() -> None:
    report = coverage.analyze_transformer_block_linear_coverage(
        _Transformer(),
        linear_cls=nn.Linear,
    )

    assert report["status"] == "passed"
    assert report["target_policy"] == "qwen_block_linears_840"
    assert report["target_count"] == 840
    assert report["non_block_exclusion_count"] == 4
    assert report["total_linear_count"] == 844
    assert report["target_role_counts"]["attn.to_q"] == 60
    assert "img_in" not in {
        record["normalized_name"] for record in report["records"] if record["is_target"]
    }
    assert report["target_block_indices"] == list(range(60))


def test_analyze_transformer_block_linear_coverage_rejects_missing_layers() -> None:
    with pytest.raises(ValueError, match="Expected 840 Qwen block Linear targets"):
        coverage.analyze_transformer_block_linear_coverage(
            _Transformer(num_layers=59),
            linear_cls=nn.Linear,
        )


def test_analyze_transformer_block_linear_coverage_rejects_extra_non_target_linear() -> None:
    transformer = _Transformer()
    transformer.extra = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="unexpected_non_target_linears"):
        coverage.analyze_transformer_block_linear_coverage(
            transformer,
            linear_cls=nn.Linear,
        )
