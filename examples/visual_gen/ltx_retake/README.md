# LTX-2.3 Native Retake

This directory is the customer runbook for the native TensorRT-LLM LTX-2.3
retake workflow. Given a prepared `retake_input.mp4` and a half-open retake
window `[start, end)`, it produces a complete `retake_output.mp4` with native
TensorRT-LLM VAE encode/decode and LTX diffusion. The bundled default prompt
uses precomputed native Gemma + LTX-connector conditioning; a custom prompt
loads Gemma and encodes it at runtime. It does not import the third-party LTX
pipeline.

The three bundled inputs exercise delete-disfluency, add-word, and replace-word
retake. `run_recipes.sh` selects their input video and timestamps by workflow
name.

> **Scope:** this package starts at a prepared `retake_input.mp4`. ASR, word
> alignment, TTS, audio insertion/deletion, and construction of a retake input
> from arbitrary source media are upstream preprocessing steps and are not
> implemented here.

## Supported configurations

| GPU | Setup script | Recipes | Customer status |
|---|---|---|---|
| H200 141 GB (SM90) | `setup_hopper_env.sh` | BF16 | BF16 E2E validated with all three bundled inputs. |
| H100 80 GB (SM90) | `setup_hopper_env.sh` | BF16 | Environment is compatible, but full-resolution 22B retake can exceed 80 GB. The bundled 1080p add/replace cases are not qualified on H100. |
| RTX PRO 6000 Blackwell (SM120) | `setup_sm120_env.sh` | BF16, FP8, NVFP4 | SM120 path; NVFP4 uses the pinned FlashInfer build. Confirm memory capacity for the chosen input. |

The default recipe is BF16. FP8 and NVFP4 are opt-in and should only be used
after the SM120 setup. Pixel equality with the upstream implementation is not
expected; the delete-disfluency upstream golden is evaluated with LPIPS.
Each invocation is a single-GPU pipeline; use `NV_GPU` at container creation
time to select which GPU is visible.

## Prerequisites

- A Linux x86-64 host with Docker, NVIDIA Container Toolkit, and a supported
  NVIDIA driver.
- Access to the pinned TensorRT-LLM rc22 image in NGC. Authenticate with
  `docker login nvcr.io` before running the start script.
- This complete TensorRT-LLM checkout. Run commands from its repository root.
- The approved LTX-2.3 22B distilled checkpoint and TalkVid identity LoRA on
  the host. A Gemma
  text-encoder bundle is additionally required for custom prompts or as a
  fallback when the bundled prompt-conditioning cache is unavailable.
  This package intentionally does not download model weights.
- Internet access to the package index during setup. The SM120 setup also needs
  GitHub access to build pinned FlashInfer PR #4272. An air-gapped FlashInfer
  source archive can be supplied with `FLASHINFER_SHARED_ARCHIVE`.
- Git LFS content available for the bundled retake inputs, upstream golden
  references, and default prompt conditioning:
  `git lfs pull --include='examples/visual_gen/ltx_retake/*_retake_input.mp4,examples/visual_gen/ltx_retake/golden_reference/*.mp4,examples/visual_gen/ltx_retake/default_prompt_conditioning.safetensors'`.

A default-prompt run needs the LTX checkpoint and TalkVid LoRA. A
custom-prompt layout is:

```text
/absolute/path/to/models/
├── ltx-2.3-22b-distilled.safetensors
├── talkvid-id-lora.safetensors
└── gemma/
    ├── config.json
    ├── model*.safetensors
    └── tokenizer files
```

The tested LTX checkpoint SHA-256 is
`b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`.
The tested TalkVid LoRA SHA-256 is
`e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9`.

## Quick start: H200 BF16

Set absolute host paths before starting the container. The start script checks
the paths, mounts them at the same locations in the container, and passes the
model variables into the container.

```bash
cd /absolute/path/to/TensorRT-LLM

export LTX_CHECKPOINT=/absolute/path/to/models/ltx-2.3-22b-distilled.safetensors
export LTX_LORA=/absolute/path/to/models/talkvid-id-lora.safetensors
RETAKE_DIR="$PWD/examples/visual_gen/ltx_retake"

docker login nvcr.io
bash "${RETAKE_DIR}/start_container.sh"

docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
  "${RETAKE_DIR}/setup_hopper_env.sh"
```

Run all three prepared workflows. With no recipe argument,
`run_recipes.sh` runs BF16 only:

```bash
for workflow in delete_disfluency add_word replace_word; do
  docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
    "${RETAKE_DIR}/run_recipes.sh" "${workflow}"
done
```

Each workflow has its own output path, so later runs do not overwrite earlier
ones:

```text
/tmp/ltx_retake/delete_disfluency/retake_bf16.mp4
/tmp/ltx_retake/add_word/retake_bf16.mp4
/tmp/ltx_retake/replace_word/retake_bf16.mp4
```

Copy the results to the host:

```bash
mkdir -p outputs/{delete_disfluency,add_word,replace_word}
for workflow in delete_disfluency add_word replace_word; do
  docker cp \
    "ltx_retake:/tmp/ltx_retake/${workflow}/retake_bf16.mp4" \
    "outputs/${workflow}/retake_output.mp4"
done
```

## Python hot API

For service or hot-latency benchmarking, create one persistent retake engine and
reuse it across requests. The first `retake()` call can be used as warmup; later
calls use the same loaded model and pipeline.

```python
from tensorrt_llm._torch.visual_gen.models.ltx2 import LTX2RetakeEngine

trtllm_engine = LTX2RetakeEngine.from_pretrained(
    checkpoint="/path/to/models/ltx-2.3-22b-distilled.safetensors",
    text_encoder="/path/to/models/gemma",
    prompt_conditioning_cache="/path/to/default_prompt_conditioning.safetensors",
    lora="/path/to/models/talkvid-id-lora.safetensors",
    quant_algo="NVFP4",
    nvfp4_attn=True,
    fp8_linear_steps=[4, 7],
)

trtllm_engine.retake(
    source="retake_input.mp4",
    output="/tmp/warmup_retake_output.mp4",
    start=2.9667,
    end=3.9333,
)
result = trtllm_engine.retake(
    source="retake_input.mp4",
    output="retake_output.mp4",
    start=2.9667,
    end=3.9333,
)
print(result.stage_timings)
```

`run_ltx2_retake()` remains available as a one-shot convenience wrapper for CLI
and simple scripts.

## Blackwell SM120 recipes

Start the container with the same model variables, but run the SM120 setup:

```bash
docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
  "${RETAKE_DIR}/setup_sm120_env.sh"
```

Then request the recipes explicitly. The tuned NVFP4 recipe enables SM120
NVFP4 attention and uses FP8 at diffusion steps 4 and 7:

```bash
docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
  "${RETAKE_DIR}/run_recipes.sh" \
  delete_disfluency bf16 fp8 nvfp4
```

Outputs are named `retake_bf16.mp4`, `retake_fp8.mp4`, and
`retake_nvfp4.mp4` in the workflow output directory. `run_nvfp4.sh` runs the
tuned NVFP4 recipe for all three prepared E2E workflows by default:

```bash
docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
  "${RETAKE_DIR}/run_nvfp4.sh"
```

Pass one or more workflow names to run a subset, for example
`run_nvfp4.sh add_word replace_word`. Each case writes to
`/tmp/ltx_retake/<workflow>/retake_nvfp4.mp4` with its own `nvfp4.log`.

For an air-gapped FlashInfer source archive, mount the archive's parent with
`LTX_MOUNTS` when starting the container, then pass its in-container path to
the setup command:

```bash
docker exec -u "$(id -u):$(id -g)" \
  -e FLASHINFER_SHARED_ARCHIVE=/absolute/mounted/path/flashinfer.tar.gz \
  ltx_retake bash "${RETAKE_DIR}/setup_sm120_env.sh"
```

## Run a customer-prepared input

The input must already be a retake input, including the intended audio edit and
duration. Mount its host directory when the container is first created:

```bash
export LTX_MOUNTS=/absolute/path/to/customer_media
bash "${RETAKE_DIR}/start_container.sh"
```

After running the appropriate setup script, pass the source path and both
timestamps. Custom inputs intentionally have no default timestamps:

```bash
docker exec -u "$(id -u):$(id -g)" \
  -e LTX_START=3.0 \
  -e LTX_END=4.5 \
  -e LTX_PROMPT='a person talking to the camera, natural head motion, clear speech' \
  -e OUT_DIR=/tmp/ltx_retake/customer_case \
  ltx_retake bash "${RETAKE_DIR}/run_recipes.sh" \
  /absolute/path/to/customer_media/retake_input.mp4 bf16

docker cp \
  ltx_retake:/tmp/ltx_retake/customer_case/retake_bf16.mp4 \
  ./retake_output.mp4
```

`LTX_START` must be smaller than `LTX_END`; both are measured in seconds and
describe the half-open interval `[start, end)` in this specific input video.

The command above uses the bundled default-prompt conditioning and does not
load Gemma. To use a custom prompt, set and mount Gemma when first starting the
container. If the default-prompt container already exists, recreate it
explicitly with `LTX_REPLACE_CONTAINER=1`:

```bash
export LTX_TEXT_ENCODER=/absolute/path/to/models/gemma
export LTX_REPLACE_CONTAINER=1  # omit this for the initial container start
bash "${RETAKE_DIR}/start_container.sh"
unset LTX_REPLACE_CONTAINER

docker exec -u "$(id -u):$(id -g)" \
  -e LTX_PROMPT='a presenter explaining a product to the camera' \
  -e LTX_START=3.0 -e LTX_END=4.5 \
  -e OUT_DIR=/tmp/ltx_retake/custom_prompt \
  ltx_retake bash "${RETAKE_DIR}/run_recipes.sh" \
  /absolute/path/to/customer_media/retake_input.mp4 bf16
```

If the default cache is missing or does not match the checkpoint, the runner
falls back to Gemma when `LTX_TEXT_ENCODER` is available; otherwise it fails
before loading the diffusion transformer.

## Rebuild the default prompt conditioning

The checked-in cache is tied to the tested checkpoint SHA-256 above. Rebuild it
only when changing the default prompt, Gemma bundle, tokenizer, or LTX
checkpoint. Run on a CUDA GPU with enough memory for Gemma and the text
connectors:

```bash
python3 "${RETAKE_DIR}/export_prompt_conditioning.py" \
  --checkpoint "${LTX_CHECKPOINT}" \
  --text-encoder "${LTX_TEXT_ENCODER}" \
  --text-encoder-id google/gemma-3-12b-it-qat-q4_0-unquantized \
  --checkpoint-sha256 b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc \
  --output "${RETAKE_DIR}/default_prompt_conditioning.safetensors"
```

The exporter saves `video_embeds`, `audio_embeds`, and `connector_mask`, then
reloads them and requires bitwise equality before succeeding. Runtime cache
selection uses a fast checkpoint fingerprint (shard sizes and safetensors
configuration metadata); the full SHA-256 remains recorded for audit without
re-reading the 46 GB checkpoint at every invocation.

## Upstream-golden LPIPS evaluation

`golden_reference/` contains the three `retake_output.mp4` files produced by the
unmodified `LTX2.3-eval` upstream `RetakePipeline`. They use its default
`fp8-cast` policy, ID-LoRA, seed 42, and the same prepared inputs and timestamps
as the native recipes. See `golden_reference/manifest.json` for hashes and exact
provenance.

After running BF16, evaluate any prepared workflow over only its regenerated
bridge window:

```bash
for workflow in delete_disfluency add_word replace_word; do
  docker exec -u "$(id -u):$(id -g)" ltx_retake bash \
    "${RETAKE_DIR}/lpips_eval.sh" "${workflow}"
done
```

The bridge windows are `[89, 118)`, `[90, 135)`, and `[38, 75)` for delete,
add, and replace respectively. Results are written to each workflow output
directory as `lpips_results.json`. If FP8 or NVFP4 output is also present, the
same result file includes BF16-versus-quantized comparisons.

## Prepared inputs

| Workflow | Input | Golden reference | Frames / resolution | Retake window |
|---|---|---|---|---|
| `delete_disfluency` | `delete_disfluency_retake_input.mp4` | `golden_reference/delete_disfluency_retake_output.mp4` | 209 / 704×1280 at 30 fps | `[2.9667, 3.9333)` |
| `add_word` | `add_word_retake_input.mp4` | `golden_reference/add_word_retake_output.mp4` | 225 / 1920×1088 at 29.97 fps | `[3.0030, 4.5045)` |
| `replace_word` | `replace_word_retake_input.mp4` | `golden_reference/replace_word_retake_output.mp4` | 113 / 1920×1088 at 29.97 fps | `[1.2679333, 2.5025)` |

`retake_inputs.json` records the source cases, media metadata, timestamps, and
SHA-256 checksums. The add/replace inputs use the first matching cases from
`LTX2.3-eval/script_editing/eval/eval_cases_subset.json`.

## Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `LTX_CHECKPOINT` | Absolute LTX checkpoint path | Required |
| `LTX_LORA` | Absolute TalkVid identity-LoRA path; fused at strength 1.0 | Required |
| `LTX_PROMPT_CONDITIONING` | Absolute default-prompt conditioning cache | Bundled cache |
| `LTX_TEXT_ENCODER` | Absolute Gemma directory | Custom prompts/fallback only |
| `LTX_CONTAINER_NAME` | Container name | `ltx_retake` |
| `LTX_IMAGE` | Locally built delivery image tag | `ltx-retake:rc22` |
| `LTX_BASE_IMAGE` | Base container reference | Pinned tested rc22 digest |
| `LTX_MOUNTS` | Extra space-separated absolute host paths mounted in place | Empty |
| `NV_GPU` | Value for `NVIDIA_VISIBLE_DEVICES` | `all` |
| `LTX_REPLACE_CONTAINER` | Replace an existing same-name container when set to `1` | `0` |
| `LTX_PROMPT` | Retake generation prompt | Talking-person prompt |
| `LTX_START`, `LTX_END` | Custom input retake window in seconds | Required for custom input |
| `OUT_DIR` | Container output directory | `/tmp/ltx_retake/<workflow>` |

`LTX_MOUNTS` is space-separated, so its path entries must not contain
whitespace. For an explicit single GPU, set (for example) `NV_GPU=0` before
running `start_container.sh`.

## Container lifecycle and troubleshooting

The start script does not silently delete an existing container. Reuse the
existing `ltx_retake` container, remove it explicitly, or opt in to replacement
with `LTX_REPLACE_CONTAINER=1`.

```bash
docker stop ltx_retake
docker rm ltx_retake
```

Common failures:

- **NGC pull denied:** authenticate to `nvcr.io` with an account entitled to
  the pinned TensorRT-LLM image.
- **Model path error:** set `LTX_CHECKPOINT`, `LTX_LORA`, and, for custom
  prompts, `LTX_TEXT_ENCODER` to absolute existing host paths before starting
  the container. They are validated before any build.
- **Prompt-conditioning mismatch:** fetch the Git LFS cache matching the tested
  checkpoint, rebuild it with the exporter, or provide `LTX_TEXT_ENCODER` for
  live Gemma fallback.
- **CUDA out of memory:** use H200 for the bundled full-resolution BF16 cases,
  reduce input resolution in upstream preprocessing, or use a validated SM120
  quantized recipe.
- **FlashInfer build cannot fetch sources:** allow GitHub access or provide a
  mounted `FLASHINFER_SHARED_ARCHIVE`.
- **LPIPS golden is invalid or missing:** fetch the Git LFS objects under
  `examples/visual_gen/ltx_retake/golden_reference/` on the host.

## Files

| File | Role |
|---|---|
| `start_container.sh` | Validate/mount models, build the pinned delivery image, and start a detached container. |
| `setup_hopper_env.sh` | Install the native overlay/dependencies and preflight Hopper SM90. |
| `setup_sm120_env.sh` | Install the native overlay/dependencies, build pinned FlashInfer, and preflight Blackwell SM120. |
| `run_recipes.sh` | Run a prepared workflow or customer-prepared input with selected recipes. |
| `run_nvfp4.sh` | Run the tuned SM120 NVFP4 recipe for all or selected prepared workflows. |
| `golden_reference/` | Upstream `LTX2.3-eval` retake outputs and provenance manifest for all three prepared workflows. |
| `export_prompt_conditioning.py` | Rebuild and validate the default-prompt conditioning cache. |
| `default_prompt_conditioning.safetensors` | Git LFS artifact containing the tested default prompt's post-connector tensors. |
| `lpips_eval.sh` | Compare any bundled workflow output with its upstream golden. |
| `Dockerfile`, `entrypoint.sh` | Thin image layer and host-user entrypoint. |
