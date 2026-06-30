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

"""Prepare Qwen-Image-Layered MXFP8 K-sweep configs and coverage artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean local envs.
    yaml = None

DEFAULT_TOTAL_BLOCKS = 60
DEFAULT_K_VALUES = (12, 8, 4, 0)
REFERENCE_VARIANT = "bf16"
NON_BACKBONE_IGNORE = ("img_in", "txt_in", "norm_out", "proj_out")
QUANT_ALGO = "FP8_BLOCK_SCALES"
REMOTE_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20"
REMOTE_CHECKOUT = "/home/scratch.minyu_gpu/project/shopee/TensorRT-LLM"
BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)(?:\.|$)")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_scalar(value: str) -> str | int | float | bool | None:
    if value == "":
        return ""
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _yaml_tokens(text: str) -> list[tuple[int, str]]:
    tokens = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue
        tokens.append((len(line) - len(line.lstrip(" ")), line.lstrip(" ")))
    return tokens


def _parse_yaml_block(tokens: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(tokens):
        return {}, index

    current_indent, current_text = tokens[index]
    if current_indent < indent:
        return {}, index

    if current_text.startswith("- "):
        values = []
        while index < len(tokens):
            item_indent, item_text = tokens[index]
            if item_indent != indent or not item_text.startswith("- "):
                break
            values.append(_parse_scalar(item_text[2:].strip()))
            index += 1
        return values, index

    mapping = {}
    while index < len(tokens):
        item_indent, item_text = tokens[index]
        if item_indent < indent:
            break
        if item_indent > indent:
            raise ValueError(f"Unexpected YAML indentation near: {item_text}")
        if ":" not in item_text:
            raise ValueError(f"Expected YAML key/value pair near: {item_text}")
        key, value = item_text.split(":", maxsplit=1)
        value = value.strip()
        index += 1
        if value:
            mapping[key] = _parse_scalar(value)
            continue
        nested, index = _parse_yaml_block(tokens, index, indent + 2)
        mapping[key] = nested
    return mapping, index


def _safe_load_yaml(text: str) -> Any:
    if yaml is not None:
        return yaml.safe_load(text)
    loaded, index = _parse_yaml_block(_yaml_tokens(text), 0, 0)
    if index != len(_yaml_tokens(text)):
        raise ValueError("Failed to consume all YAML tokens")
    return loaded


def _format_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def _dump_simple_yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        return [f"{prefix}- {_format_scalar(item)}" for item in value]
    return [f"{prefix}{_format_scalar(value)}"]


def _safe_dump_yaml(payload: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(payload, sort_keys=False)
    return "\n".join(_dump_simple_yaml(payload)) + "\n"


def read_yaml(path: Path) -> dict[str, Any]:
    loaded = _safe_load_yaml(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_dump_yaml(payload), encoding="utf-8")


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def variant_name(k_edge_bf16: int) -> str:
    return f"k{k_edge_bf16}"


def parse_k_values(values: list[str] | None) -> list[int]:
    if values is None:
        return list(DEFAULT_K_VALUES)
    parsed = [int(value) for value in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"K values must be unique: {parsed}")
    return parsed


def build_edge_ignore_list(
    k_edge_bf16: int,
    *,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
    non_backbone_ignore: tuple[str, ...] = NON_BACKBONE_IGNORE,
) -> list[str]:
    if k_edge_bf16 < 0:
        raise ValueError(f"k_edge_bf16 must be non-negative, got {k_edge_bf16}")
    if total_blocks <= 0:
        raise ValueError(f"total_blocks must be positive, got {total_blocks}")
    if k_edge_bf16 * 2 > total_blocks:
        raise ValueError(
            f"k_edge_bf16={k_edge_bf16} overlaps for {total_blocks} transformer blocks"
        )

    block_indices = list(range(k_edge_bf16))
    block_indices.extend(range(total_blocks - k_edge_bf16, total_blocks))
    return [*non_backbone_ignore, *[f"transformer_blocks.{idx}" for idx in block_indices]]


def materialize_quant_config(
    template: dict[str, Any],
    *,
    k_edge_bf16: int,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
) -> dict[str, Any]:
    config = deepcopy(template)
    quant_config = dict(config.get("quant_config") or {})
    quant_config["quant_algo"] = QUANT_ALGO
    quant_config["dynamic"] = True
    quant_config["ignore"] = build_edge_ignore_list(
        k_edge_bf16,
        total_blocks=total_blocks,
    )
    config["quant_config"] = quant_config
    return config


def materialize_bf16_config(template: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(template)
    config.pop("quant_config", None)
    return config


def materialize_configs(
    *,
    template_path: Path,
    output_dir: Path,
    k_values: list[int],
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
) -> dict[str, Path]:
    template = read_yaml(template_path)
    paths: dict[str, Path] = {}

    bf16_path = output_dir / "qwen-image-layered-bf16.yaml"
    write_yaml(bf16_path, materialize_bf16_config(template))
    paths[REFERENCE_VARIANT] = bf16_path

    for k_edge_bf16 in k_values:
        name = variant_name(k_edge_bf16)
        path = output_dir / f"qwen-image-layered-{name}-mxfp8.yaml"
        config = materialize_quant_config(
            template,
            k_edge_bf16=k_edge_bf16,
            total_blocks=total_blocks,
        )
        write_yaml(path, config)
        paths[name] = path

    return paths


def module_matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def block_index(name: str) -> int | None:
    match = BLOCK_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def preserved_reason(name: str, ignore: list[str]) -> str | None:
    for entry in ignore:
        if module_matches_prefix(name, entry):
            return entry
    return None


def module_role(name: str) -> str:
    if any(module_matches_prefix(name, entry) for entry in NON_BACKBONE_IGNORE):
        return "non_backbone"
    if block_index(name) is not None:
        return "transformer_block"
    return "other"


def normalize_linear_names(payload: Any) -> list[str]:
    if isinstance(payload, list):
        names = payload
    elif isinstance(payload, dict) and isinstance(payload.get("linear_names"), list):
        names = payload["linear_names"]
    else:
        raise ValueError("Expected a list of linear names or {'linear_names': [...]}")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("All linear names must be strings")
    return sorted(set(names))


def inspect_model_linear_names(model_config_path: Path, config_path: Path) -> list[str]:
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        QwenImageTransformer2DModel,
    )
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import AttentionConfig

    transformer_cfg = read_json(model_config_path)
    visual_gen_cfg = read_yaml(config_path)
    quant_cfg_dict = visual_gen_cfg.get("quant_config")
    if quant_cfg_dict:
        quant_config, quant_config_dict, dynamic_weight_quant, _ = (
            DiffusionPipelineConfig.load_diffusion_quant_config(quant_cfg_dict)
        )
    else:
        quant_config = QuantConfig()
        quant_config_dict = None
        dynamic_weight_quant = False

    attention_config = AttentionConfig(**visual_gen_cfg.get("attention_config", {}))
    model_config = DiffusionModelConfig(
        skip_create_weights_in_init=True,
        dynamic_weight_quant=dynamic_weight_quant,
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
        attention=attention_config,
    )
    model = QwenImageTransformer2DModel.from_config_dict(
        transformer_cfg,
        model_config=model_config,
        attn_backend=attention_config.backend,
    )
    if hasattr(model, "apply_quant_config_exclude_modules"):
        model.apply_quant_config_exclude_modules()
    elif quant_config.exclude_modules:
        no_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for name, module in model.named_modules():
            if not isinstance(module, Linear) or getattr(module, "quant_config", None) is None:
                continue
            if quant_config.is_module_excluded_from_quantization(name):
                module.quant_config = no_quant_config
    return sorted(name for name, module in model.named_modules() if isinstance(module, Linear))


def build_coverage(
    *,
    variant: str,
    config_path: Path,
    config: dict[str, Any],
    linear_names: list[str],
    git_sha: str | None,
    total_blocks: int = DEFAULT_TOTAL_BLOCKS,
) -> dict[str, Any]:
    quant_config = config.get("quant_config") or {}
    ignore = list(quant_config.get("ignore") or [])
    quant_algo = quant_config.get("quant_algo")
    is_quantized_variant = quant_algo == QUANT_ALGO
    per_block = {
        idx: {"block_index": idx, "bf16_preserved_linear": 0, "mxfp8_linear": 0}
        for idx in range(total_blocks)
    }
    modules = []
    counts = {
        "linear_total": 0,
        "mxfp8_linear": 0,
        "bf16_preserved_linear": 0,
        "non_backbone_linear": 0,
        "other_linear": 0,
    }

    for name in linear_names:
        reason = preserved_reason(name, ignore) if is_quantized_variant else "bf16_reference"
        expected = "bf16" if reason is not None else "mxfp8"
        role = module_role(name)
        idx = block_index(name)
        counts["linear_total"] += 1
        if expected == "mxfp8":
            counts["mxfp8_linear"] += 1
            if idx is not None and idx in per_block:
                per_block[idx]["mxfp8_linear"] += 1
        else:
            counts["bf16_preserved_linear"] += 1
            if idx is not None and idx in per_block:
                per_block[idx]["bf16_preserved_linear"] += 1
        if role == "non_backbone":
            counts["non_backbone_linear"] += 1
        elif role == "other":
            counts["other_linear"] += 1

        modules.append(
            {
                "name": name,
                "block_index": idx,
                "role": role,
                "expected": expected,
                "preserved_reason": reason,
            }
        )

    k_edge_bf16 = int(variant[1:]) if variant.startswith("k") and variant[1:].isdigit() else None
    return {
        "schema_version": 1,
        "variant": variant,
        "k_edge_bf16": k_edge_bf16,
        "config_path": str(config_path),
        "git_commit": git_sha,
        "quant_algo": quant_algo,
        "attention_config": config.get("attention_config") or {},
        "ignore": ignore,
        "counts": counts,
        "linear_modules": modules,
        "by_block": [per_block[idx] for idx in range(total_blocks)],
    }


def write_manifest(
    *,
    samples_path: Path,
    config_paths: dict[str, Path],
    output_json: Path,
    output_root: Path,
    model: str,
    artifact_root: Path,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    samples = read_json(samples_path)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected non-empty sample list in {samples_path}")

    variants = []
    for name, config_path in config_paths.items():
        artifact_paths = {
            sample["id"]: str(artifact_root / name / f"{sample['id']}.pt") for sample in samples
        }
        metadata: dict[str, Any] = {"preset": name}
        if name.startswith("k") and name[1:].isdigit():
            metadata["k_edge_bf16"] = int(name[1:])
        variants.append(
            {
                "name": name,
                "model": model,
                "visual_gen_args": str(config_path),
                "checkpoint_path": checkpoint_path,
                "artifact_paths": artifact_paths,
                "metadata": metadata,
            }
        )

    manifest = {
        "output_root": str(output_root),
        "samples": samples,
        "variants": variants,
    }
    write_json(output_json, manifest)
    return manifest


def write_provenance(
    *,
    output_json: Path,
    run_root: Path,
    checkout: str,
    allocation: str,
    node: str,
    docker_image: str,
    manifest_path: Path,
    metrics_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    command = (
        f"ssh-gw exec {allocation} --no-enroot "
        f"'cd {checkout} && docker run --rm --gpus all --ipc=host "
        f"-v {checkout}:/workspace/TensorRT-LLM -v {run_root}:{run_root} "
        f"-w /workspace/TensorRT-LLM {docker_image} "
        f"python3 scripts/visualgen_eval/qwen_image_layered_quality.py "
        f"--manifest {manifest_path} --output-json {metrics_path} "
        f"--artifact-format pt --save-audit-pngs'"
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "checkout": checkout,
        "allocation": allocation,
        "node": node,
        "docker_image": docker_image,
        "git_commit": git_commit(project_root),
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "quality_command": command,
    }
    write_json(output_json, payload)
    return payload


def _materialize_configs_cmd(args: argparse.Namespace) -> None:
    paths = materialize_configs(
        template_path=args.template,
        output_dir=args.output_dir,
        k_values=parse_k_values(args.k),
        total_blocks=args.total_blocks,
    )
    write_json(
        args.output_dir / "config-index.json", {name: str(path) for name, path in paths.items()}
    )


def _write_manifest_cmd(args: argparse.Namespace) -> None:
    config_paths = {
        REFERENCE_VARIANT: args.config_dir / "qwen-image-layered-bf16.yaml",
        **{
            variant_name(k): args.config_dir / f"qwen-image-layered-{variant_name(k)}-mxfp8.yaml"
            for k in parse_k_values(args.k)
        },
    }
    missing = [str(path) for path in config_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing generated configs: {missing}")
    write_manifest(
        samples_path=args.samples_json,
        config_paths=config_paths,
        output_json=args.output_json,
        output_root=args.output_root,
        model=args.model,
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_path,
    )


def _inspect_coverage_cmd(args: argparse.Namespace) -> None:
    variants = [REFERENCE_VARIANT, *[variant_name(k) for k in parse_k_values(args.k)]]
    if args.linear_names_json is not None:
        linear_names = normalize_linear_names(read_json(args.linear_names_json))
    elif args.model_config is not None:
        first_config = args.config_dir / "qwen-image-layered-bf16.yaml"
        linear_names = inspect_model_linear_names(args.model_config, first_config)
    else:
        raise ValueError("Either --linear-names-json or --model-config is required")

    git_sha = git_commit(args.project_root)
    for variant in variants:
        if variant == REFERENCE_VARIANT:
            config_path = args.config_dir / "qwen-image-layered-bf16.yaml"
        else:
            config_path = args.config_dir / f"qwen-image-layered-{variant}-mxfp8.yaml"
        config = read_yaml(config_path)
        coverage = build_coverage(
            variant=variant,
            config_path=config_path,
            config=config,
            linear_names=linear_names,
            git_sha=git_sha,
            total_blocks=args.total_blocks,
        )
        write_json(args.output_dir / f"{variant}.coverage.json", coverage)


def _write_provenance_cmd(args: argparse.Namespace) -> None:
    write_provenance(
        output_json=args.output_json,
        run_root=args.run_root,
        checkout=args.checkout,
        allocation=args.allocation,
        node=args.node,
        docker_image=args.docker_image,
        manifest_path=args.manifest,
        metrics_path=args.metrics,
        project_root=args.project_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-configs")
    materialize.add_argument("--template", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--k", nargs="*")
    materialize.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS)
    materialize.set_defaults(func=_materialize_configs_cmd)

    manifest = subparsers.add_parser("write-manifest")
    manifest.add_argument("--samples-json", type=Path, required=True)
    manifest.add_argument("--config-dir", type=Path, required=True)
    manifest.add_argument("--output-json", type=Path, required=True)
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--artifact-root", type=Path, required=True)
    manifest.add_argument("--model", required=True)
    manifest.add_argument("--checkpoint-path")
    manifest.add_argument("--k", nargs="*")
    manifest.set_defaults(func=_write_manifest_cmd)

    coverage = subparsers.add_parser("inspect-coverage")
    coverage.add_argument("--config-dir", type=Path, required=True)
    coverage.add_argument("--output-dir", type=Path, required=True)
    coverage.add_argument("--linear-names-json", type=Path)
    coverage.add_argument("--model-config", type=Path)
    coverage.add_argument("--project-root", type=Path, default=Path.cwd())
    coverage.add_argument("--k", nargs="*")
    coverage.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS)
    coverage.set_defaults(func=_inspect_coverage_cmd)

    provenance = subparsers.add_parser("write-provenance")
    provenance.add_argument("--output-json", type=Path, required=True)
    provenance.add_argument("--run-root", type=Path, required=True)
    provenance.add_argument("--checkout", default=REMOTE_CHECKOUT)
    provenance.add_argument("--allocation", default="sc-2886562")
    provenance.add_argument("--node", default="viking-dvt-151")
    provenance.add_argument("--docker-image", default=REMOTE_IMAGE)
    provenance.add_argument("--manifest", type=Path, required=True)
    provenance.add_argument("--metrics", type=Path, required=True)
    provenance.add_argument("--project-root", type=Path, default=Path.cwd())
    provenance.set_defaults(func=_write_provenance_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
