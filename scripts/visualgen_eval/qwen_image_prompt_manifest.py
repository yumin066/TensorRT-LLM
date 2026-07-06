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

"""Build the ordinary Qwen-Image prompt calibration manifest for MXFP8 QAT."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

PromptSplit = Literal["smoke", "fast_calibration", "main_calibration", "held_out"]
ArtifactPolicy = Literal["durable_b300_enroot_only", "local_smoke"]

MANIFEST_FORMAT = "qwen_image_prompt_manifest_v1"
SUMMARY_FORMAT = "qwen_image_prompt_manifest_summary_v1"
DEFAULT_ARTIFACT_POLICY: ArtifactPolicy = "durable_b300_enroot_only"
LOCAL_SMOKE_ARTIFACT_POLICY: ArtifactPolicy = "local_smoke"
DEFAULT_MODEL = "Qwen/Qwen-Image"
DEFAULT_CLUSTER_ALIAS = "B300-mars"
DEFAULT_CONTAINER_RUNTIME = "enroot"
DEFAULT_ENROOT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20"
DEFAULT_CHECKOUT_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/ssh-gw/TensorRT-LLM"
)
DEFAULT_RUN_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/ssh-gw/"
    "TensorRT-LLM/runs/qwen_image_qat_data/qwen_image_prompt_calibration_v1"
)
DEFAULT_CACHE_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_devtech_all/minyu/.cache/huggingface"
)
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted text, watermark, extra fingers, bad anatomy"
)
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_SEED_BASE = 2_026_070_600
DEFAULT_CFG_BRANCHES = ("cond", "negative")
DEFAULT_SPLIT_COUNTS: dict[PromptSplit, int] = {
    "smoke": 1,
    "fast_calibration": 32,
    "main_calibration": 128,
    "held_out": 32,
}
CALIBRATION_SPLITS = {"smoke", "fast_calibration", "main_calibration"}
REQUIRED_CATEGORIES = {
    "photorealistic_scene",
    "person_portrait",
    "object_product",
    "indoor_outdoor",
    "text_rendering",
    "style_transfer",
    "complex_composition",
    "high_frequency_texture",
    "color_lighting_variation",
}
REQUIRED_RECORD_FIELDS = {
    "prompt_id",
    "split",
    "source",
    "categories",
    "prompt",
    "negative_prompt",
    "seed",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "max_sequence_length",
    "model",
    "cfg_branches",
    "expected_tuple_count",
}
REQUIRED_CAPTURE_FIELDS = (
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "img_shapes",
    "txt_seq_lens",
    "target_output",
    "prompt_id",
    "timestep_index",
    "cfg_branch",
    "timestep_bin",
)
REQUIRED_DURABLE_PATH_FIELDS = (
    "checkout_root",
    "run_root",
    "cache_root",
    "prompt_manifest",
    "bf16_references",
    "teacher_tuples",
)
RUN_ROOT_DURABLE_PATH_FIELDS = (
    "prompt_manifest",
    "bf16_references",
    "teacher_tuples",
)


@dataclass(frozen=True)
class PromptTemplate:
    split: PromptSplit
    source: str
    categories: tuple[str, ...]
    prompt: str


CATEGORY_LIBRARY: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "photorealistic_scene",
        (
            "a photorealistic city street after rain",
            "a wide-angle mountain village at sunrise",
            "a documentary-style coastal market at dusk",
        ),
    ),
    (
        "person_portrait",
        (
            "a natural portrait of a young chef in a bright kitchen",
            "a cinematic portrait of an engineer beside a workbench",
            "a candid portrait of a dancer tying shoes backstage",
        ),
    ),
    (
        "object_product",
        (
            "a studio product photo of a translucent smart speaker",
            "a detailed product render of a folding camping lantern",
            "a catalog photo of a matte ceramic tea set",
        ),
    ),
    (
        "indoor_outdoor",
        (
            "an airy reading room opening into a green courtyard",
            "a compact workshop with tools and a garden view",
            "a modern cafe patio connected to a warm interior",
        ),
    ),
    (
        "text_rendering",
        (
            "a clean poster that says QWEN IMAGE in bold letters",
            "a bakery sign with the words FRESH BREAD at the window",
            "a notebook cover titled MXFP8 QAT with neat typography",
        ),
    ),
    (
        "style_transfer",
        (
            "a watercolor interpretation of a tram crossing a bridge",
            "a paper-cut style illustration of a night garden",
            "an ink wash landscape with subtle futuristic buildings",
        ),
    ),
    (
        "complex_composition",
        (
            "a busy robotics lab with five distinct workstations",
            "a layered festival scene with lanterns, food stalls, and rain",
            "an isometric apartment building showing multiple rooms",
        ),
    ),
    (
        "high_frequency_texture",
        (
            "a close-up macro photo of woven carbon fiber",
            "a sharp textile sample with intricate embroidered patterns",
            "a field of frost crystals on dark glass",
        ),
    ),
    (
        "color_lighting_variation",
        (
            "a neon-lit alley with teal shadows and amber highlights",
            "a still life under split red and blue studio lights",
            "a greenhouse scene with dappled sunlight and deep shadows",
        ),
    ),
)

DETAIL_PHRASES = (
    "with crisp foreground detail and a soft but coherent background",
    "with multiple depth layers and small readable visual elements",
    "with balanced composition and realistic spatial relationships",
    "with reflective materials, subtle shadows, and clean edges",
    "with varied textures that stress fine detail reconstruction",
    "with saturated colors but natural contrast",
    "with a central subject and several secondary objects",
    "with both smooth gradients and high-frequency patterns",
)

LIGHTING_PHRASES = (
    "using soft morning light",
    "using dramatic side lighting",
    "using overcast natural light",
    "using warm indoor light",
    "using high contrast studio light",
    "using mixed daylight and artificial light",
)

SOURCE_CYCLE = (
    "qwenimage_custom",
    "drawbench_style",
    "partiprompts_style",
    "ms_coco_caption_style",
    "stable_diffusion_prompts_style",
)


def _split_short_name(split: PromptSplit) -> str:
    return {
        "smoke": "smoke",
        "fast_calibration": "fast",
        "main_calibration": "main",
        "held_out": "heldout",
    }[split]


def _split_offset(split: PromptSplit) -> int:
    return {
        "smoke": 0,
        "fast_calibration": 17,
        "main_calibration": 101,
        "held_out": 303,
    }[split]


def _build_prompt_template(split: PromptSplit, index: int) -> PromptTemplate:
    global_index = _split_offset(split) + index
    category, stems = CATEGORY_LIBRARY[global_index % len(CATEGORY_LIBRARY)]
    stem = stems[(global_index // len(CATEGORY_LIBRARY)) % len(stems)]
    detail = DETAIL_PHRASES[global_index % len(DETAIL_PHRASES)]
    lighting = LIGHTING_PHRASES[(global_index // len(DETAIL_PHRASES)) % len(LIGHTING_PHRASES)]
    prompt = f"{stem}, {detail}, {lighting}."
    source = SOURCE_CYCLE[global_index % len(SOURCE_CYCLE)]
    categories = _categories_for(category, global_index)
    return PromptTemplate(split=split, source=source, categories=categories, prompt=prompt)


def _categories_for(primary_category: str, global_index: int) -> tuple[str, ...]:
    categories = [primary_category]
    if global_index % 5 == 0 and primary_category != "color_lighting_variation":
        categories.append("color_lighting_variation")
    if global_index % 7 == 0 and primary_category != "high_frequency_texture":
        categories.append("high_frequency_texture")
    return tuple(categories)


def build_default_prompt_records(
    *,
    model: str = DEFAULT_MODEL,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed_base: int = DEFAULT_SEED_BASE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> list[dict[str, object]]:
    """Build the default 1/32/128/32 ordinary Qwen-Image prompt records."""
    records: list[dict[str, object]] = []
    prompt_index = 0
    for split, count in DEFAULT_SPLIT_COUNTS.items():
        for split_index in range(count):
            template = _build_prompt_template(split, split_index)
            prompt_id = f"qwen_image_{_split_short_name(split)}_{split_index:04d}"
            record = {
                "prompt_id": prompt_id,
                "split": split,
                "source": template.source,
                "categories": list(template.categories),
                "prompt": template.prompt,
                "negative_prompt": negative_prompt,
                "seed": seed_base + prompt_index,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "model": model,
                "cfg_branches": list(DEFAULT_CFG_BRANCHES),
                "expected_tuple_count": num_inference_steps * len(DEFAULT_CFG_BRANCHES),
                "trajectory_source": "bf16_teacher",
            }
            records.append(record)
            prompt_index += 1
    validate_prompt_records(records)
    return records


def validate_prompt_records(
    records: list[dict[str, object]],
    *,
    expected_split_counts: dict[PromptSplit, int] | None = None,
) -> dict[str, int]:
    """Validate prompt records and return observed split counts."""
    if expected_split_counts is None:
        expected_split_counts = DEFAULT_SPLIT_COUNTS
    if not records:
        raise ValueError("prompt manifest must contain at least one record")

    split_counts: Counter[str] = Counter()
    prompt_ids: set[str] = set()
    calibration_prompts: set[str] = set()
    held_out_prompts: set[str] = set()
    observed_categories: set[str] = set()
    common_settings: dict[str, object] = {}

    for record in records:
        missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
        if missing:
            raise ValueError(f"prompt record is missing required fields: {missing}")

        prompt_id = _expect_string(record, "prompt_id")
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate prompt_id in prompt manifest: {prompt_id}")
        prompt_ids.add(prompt_id)

        split = _expect_split(record)
        split_counts[split] += 1

        source = _expect_string(record, "source")
        if not source:
            raise ValueError(f"prompt {prompt_id} must include a non-empty source")

        categories = _expect_string_list(record, "categories")
        if not categories:
            raise ValueError(f"prompt {prompt_id} must include at least one category")
        observed_categories.update(categories)

        prompt = _expect_string(record, "prompt")
        if not prompt:
            raise ValueError(f"prompt {prompt_id} must include prompt text")
        normalized_prompt = " ".join(prompt.casefold().split())
        if split in CALIBRATION_SPLITS:
            calibration_prompts.add(normalized_prompt)
        else:
            held_out_prompts.add(normalized_prompt)

        _validate_diffusion_settings(record, prompt_id)
        _validate_common_settings(record, prompt_id, common_settings)

    expected_counts = dict(expected_split_counts)
    observed_counts = {split: split_counts.get(split, 0) for split in expected_counts}
    if observed_counts != expected_counts:
        raise ValueError(
            f"prompt split counts do not match expected counts: "
            f"observed={observed_counts}, expected={expected_counts}"
        )

    overlap = calibration_prompts & held_out_prompts
    if overlap:
        raise ValueError("held-out prompts must not overlap with calibration prompts")

    missing_categories = sorted(REQUIRED_CATEGORIES - observed_categories)
    if missing_categories:
        raise ValueError(f"prompt manifest is missing required categories: {missing_categories}")

    return observed_counts


def _expect_string(record: dict[str, object], field_name: str) -> str:
    value = record[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _expect_split(record: dict[str, object]) -> PromptSplit:
    split = _expect_string(record, "split")
    if split not in DEFAULT_SPLIT_COUNTS:
        raise ValueError(f"invalid prompt split: {split}")
    return cast(PromptSplit, split)


def _expect_string_list(record: dict[str, object], field_name: str) -> list[str]:
    value = record[field_name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _validate_diffusion_settings(record: dict[str, object], prompt_id: str) -> None:
    seed = record["seed"]
    height = record["height"]
    width = record["width"]
    num_inference_steps = record["num_inference_steps"]
    guidance_scale = record["guidance_scale"]
    max_sequence_length = record["max_sequence_length"]
    cfg_branches = _expect_string_list(record, "cfg_branches")
    expected_tuple_count = record["expected_tuple_count"]

    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"prompt {prompt_id} seed must be a non-negative integer")
    for field_name, value in (("height", height), ("width", width)):
        if not isinstance(value, int) or value <= 0 or value % 8 != 0:
            raise ValueError(f"prompt {prompt_id} {field_name} must be positive and divisible by 8")
    if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
        raise ValueError(f"prompt {prompt_id} num_inference_steps must be positive")
    if not isinstance(guidance_scale, (int, float)) or guidance_scale <= 0:
        raise ValueError(f"prompt {prompt_id} guidance_scale must be positive")
    if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
        raise ValueError(f"prompt {prompt_id} max_sequence_length must be positive")
    if tuple(cfg_branches) != DEFAULT_CFG_BRANCHES:
        raise ValueError(f"prompt {prompt_id} cfg_branches must be {list(DEFAULT_CFG_BRANCHES)}")
    if expected_tuple_count != num_inference_steps * len(cfg_branches):
        raise ValueError(f"prompt {prompt_id} expected_tuple_count does not match CFG tuple count")


def _validate_common_settings(
    record: dict[str, object],
    prompt_id: str,
    common_settings: dict[str, object],
) -> None:
    for field_name in (
        "model",
        "negative_prompt",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "cfg_branches",
    ):
        value = record[field_name]
        if field_name not in common_settings:
            common_settings[field_name] = value
            continue
        if common_settings[field_name] != value:
            raise ValueError(f"prompt {prompt_id} {field_name} does not match manifest settings")


def build_summary(
    *,
    records: list[dict[str, object]],
    manifest_jsonl: Path,
    checkout_root: str = DEFAULT_CHECKOUT_ROOT,
    run_root: str = DEFAULT_RUN_ROOT,
    cache_root: str = DEFAULT_CACHE_ROOT,
    cluster_alias: str = DEFAULT_CLUSTER_ALIAS,
    container_runtime: str = DEFAULT_CONTAINER_RUNTIME,
    enroot_image: str = DEFAULT_ENROOT_IMAGE,
    artifact_policy: ArtifactPolicy = DEFAULT_ARTIFACT_POLICY,
) -> dict[str, object]:
    """Build and validate the manifest summary/provenance payload."""
    split_counts = validate_prompt_records(records)
    prompt_id_digest = _prompt_id_digest(records)
    expected_tuples = sum(int(record["expected_tuple_count"]) for record in records)
    categories = sorted(
        {category for record in records for category in _expect_string_list(record, "categories")}
    )
    sources = sorted({_expect_string(record, "source") for record in records})
    payload: dict[str, object] = {
        "format": SUMMARY_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "model": records[0]["model"],
        "cluster_alias": cluster_alias,
        "container_runtime": container_runtime,
        "enroot_image": enroot_image,
        "height": records[0]["height"],
        "width": records[0]["width"],
        "num_inference_steps": records[0]["num_inference_steps"],
        "guidance_scale": records[0]["guidance_scale"],
        "max_sequence_length": records[0]["max_sequence_length"],
        "cfg_branches": list(DEFAULT_CFG_BRANCHES),
        "split_counts": split_counts,
        "prompt_count": len(records),
        "prompt_id_sha256": prompt_id_digest,
        "prompt_sources": sources,
        "prompt_categories": categories,
        "expected_tuples_per_prompt": records[0]["expected_tuple_count"],
        "expected_total_teacher_tuples": expected_tuples,
        "durable_paths": {
            "checkout_root": checkout_root,
            "run_root": run_root,
            "cache_root": cache_root,
            "prompt_manifest": f"{run_root}/manifests/qwen_image_qat_prompts_v1.jsonl",
            "bf16_references": f"{run_root}/references/bf16",
            "teacher_tuples": f"{run_root}/teacher_tuples",
        },
        "local_output_paths": {
            "manifest_jsonl": str(manifest_jsonl),
        },
        "artifact_policy": artifact_policy,
        "task2_input_ready": artifact_policy == DEFAULT_ARTIFACT_POLICY,
        "required_capture_fields": list(REQUIRED_CAPTURE_FIELDS),
    }
    validate_summary(payload, records=records)
    return payload


def validate_summary(summary: dict[str, object], *, records: list[dict[str, object]]) -> None:
    """Validate B300-mars/enroot provenance for a prompt manifest summary."""
    validate_prompt_records(records)
    _validate_summary_derived_fields(summary, records)

    if summary.get("cluster_alias") != DEFAULT_CLUSTER_ALIAS:
        raise ValueError(f"cluster_alias must be {DEFAULT_CLUSTER_ALIAS}")
    if summary.get("container_runtime") != DEFAULT_CONTAINER_RUNTIME:
        raise ValueError(f"container_runtime must be {DEFAULT_CONTAINER_RUNTIME}")
    if summary.get("enroot_image") != DEFAULT_ENROOT_IMAGE:
        raise ValueError(f"enroot_image must be {DEFAULT_ENROOT_IMAGE}")
    if _contains_forbidden_runtime(summary):
        raise ValueError("prompt manifest summary must not include Docker provenance")

    artifact_policy = summary.get("artifact_policy")
    if artifact_policy not in (DEFAULT_ARTIFACT_POLICY, LOCAL_SMOKE_ARTIFACT_POLICY):
        raise ValueError("summary artifact_policy is invalid")
    if summary.get("task2_input_ready") != (artifact_policy == DEFAULT_ARTIFACT_POLICY):
        raise ValueError("summary task2_input_ready does not match artifact_policy")

    durable_paths = summary.get("durable_paths")
    if not isinstance(durable_paths, dict):
        raise ValueError("summary durable_paths must be a mapping")
    for key in REQUIRED_DURABLE_PATH_FIELDS:
        value = durable_paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"summary durable_paths.{key} must be a non-empty string")
        if value.startswith("/tmp"):
            raise ValueError(f"summary durable_paths.{key} must not be under /tmp")
    run_root = str(durable_paths["run_root"]).rstrip("/")
    for key in RUN_ROOT_DURABLE_PATH_FIELDS:
        value = str(durable_paths[key])
        if not _path_is_under(value, run_root):
            raise ValueError(f"summary durable_paths.{key} must be under durable run_root")

    local_output_paths = summary.get("local_output_paths")
    if not isinstance(local_output_paths, dict):
        raise ValueError("summary local_output_paths must be a mapping")
    manifest_jsonl = local_output_paths.get("manifest_jsonl")
    if not isinstance(manifest_jsonl, str) or not manifest_jsonl:
        raise ValueError("summary local_output_paths.manifest_jsonl must be a non-empty string")
    if (
        artifact_policy == DEFAULT_ARTIFACT_POLICY
        and manifest_jsonl != durable_paths["prompt_manifest"]
    ):
        raise ValueError("formal prompt manifest path must match durable_paths.prompt_manifest")


def _validate_summary_derived_fields(
    summary: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    expected = _expected_summary_fields(records)
    for field_name, expected_value in expected.items():
        if summary.get(field_name) != expected_value:
            raise ValueError(f"summary {field_name} does not match prompt records")

    capture_fields = summary.get("required_capture_fields")
    if not isinstance(capture_fields, list) or set(capture_fields) != set(REQUIRED_CAPTURE_FIELDS):
        raise ValueError(
            "summary required_capture_fields must match the required capture field set"
        )


def _expected_summary_fields(records: list[dict[str, object]]) -> dict[str, object]:
    first_record = records[0]
    return {
        "format": SUMMARY_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "model": first_record["model"],
        "height": first_record["height"],
        "width": first_record["width"],
        "num_inference_steps": first_record["num_inference_steps"],
        "guidance_scale": first_record["guidance_scale"],
        "max_sequence_length": first_record["max_sequence_length"],
        "cfg_branches": list(DEFAULT_CFG_BRANCHES),
        "split_counts": validate_prompt_records(records),
        "prompt_count": len(records),
        "prompt_id_sha256": _prompt_id_digest(records),
        "prompt_sources": sorted({_expect_string(record, "source") for record in records}),
        "prompt_categories": sorted(
            {
                category
                for record in records
                for category in _expect_string_list(record, "categories")
            }
        ),
        "expected_tuples_per_prompt": first_record["expected_tuple_count"],
        "expected_total_teacher_tuples": sum(
            int(record["expected_tuple_count"]) for record in records
        ),
    }


def _path_is_under(path: str, root: str) -> bool:
    normalized_root = root.rstrip("/")
    return path == normalized_root or path.startswith(f"{normalized_root}/")


def _contains_forbidden_runtime(value: object, *, key_name: str = "") -> bool:
    if "docker" in key_name.casefold():
        return True
    if isinstance(value, str):
        return "docker" in value.casefold()
    if isinstance(value, dict):
        return any(
            _contains_forbidden_runtime(child_value, key_name=str(child_key))
            for child_key, child_value in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_runtime(child_value) for child_value in value)
    return False


def _prompt_id_digest(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_expect_string(record, "prompt_id").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    validate_prompt_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        records.append(loaded)
    validate_prompt_records(records)
    return records


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in {path}")
    return loaded


def _write_manifest_command(args: argparse.Namespace) -> None:
    manifest_jsonl = Path(args.manifest_jsonl)
    records = build_default_prompt_records(
        model=args.model,
        negative_prompt=args.negative_prompt,
        seed_base=args.seed_base,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    summary = build_summary(
        records=records,
        manifest_jsonl=manifest_jsonl,
        checkout_root=args.checkout_root,
        run_root=args.run_root,
        cache_root=args.cache_root,
        cluster_alias=args.cluster_alias,
        container_runtime=args.container_runtime,
        enroot_image=args.enroot_image,
        artifact_policy=args.artifact_policy,
    )
    write_jsonl(manifest_jsonl, records)
    write_json(Path(args.summary_json), summary)


def _validate_manifest_command(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.manifest_jsonl))
    if args.summary_json is not None:
        validate_summary(read_json(Path(args.summary_json)), records=records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write-manifest")
    write_parser.add_argument("--manifest-jsonl", required=True)
    write_parser.add_argument("--summary-json", required=True)
    write_parser.add_argument("--model", default=DEFAULT_MODEL)
    write_parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    write_parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    write_parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    write_parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    write_parser.add_argument(
        "--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    write_parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    write_parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    write_parser.add_argument("--checkout-root", default=DEFAULT_CHECKOUT_ROOT)
    write_parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    write_parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    write_parser.add_argument("--cluster-alias", default=DEFAULT_CLUSTER_ALIAS)
    write_parser.add_argument("--container-runtime", default=DEFAULT_CONTAINER_RUNTIME)
    write_parser.add_argument("--enroot-image", default=DEFAULT_ENROOT_IMAGE)
    write_parser.add_argument(
        "--artifact-policy",
        choices=(DEFAULT_ARTIFACT_POLICY, LOCAL_SMOKE_ARTIFACT_POLICY),
        default=DEFAULT_ARTIFACT_POLICY,
    )
    write_parser.set_defaults(func=_write_manifest_command)

    validate_parser = subparsers.add_parser("validate-manifest")
    validate_parser.add_argument("--manifest-jsonl", required=True)
    validate_parser.add_argument("--summary-json")
    validate_parser.set_defaults(func=_validate_manifest_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
