#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the native LTX-2 retake e2e for one or more recipes. Prepared workflow
# outputs are isolated under /tmp/ltx_retake/<workflow> by default.
#
# Usage: run_recipes.sh <workflow|source.mp4> [recipe ...]
# Workflows: delete_disfluency, add_word, replace_word.
# Recipes default to bf16. FP8/NVFP4 are opt-in Blackwell recipes.
#
# Env overrides: LTX_CHECKPOINT, LTX_TEXT_ENCODER, LTX_PROMPT, LTX_START,
#                LTX_END, OUT_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CKPT="${LTX_CHECKPOINT:-}"
GEMMA="${LTX_TEXT_ENCODER:-}"
PROMPT="${LTX_PROMPT:-a person talking to the camera, natural head motion, clear speech}"
INPUT="${1:?usage: run_recipes.sh <workflow|source.mp4> [recipe ...]}"
shift
RECIPES=("$@"); [ ${#RECIPES[@]} -eq 0 ] && RECIPES=(bf16)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[ -n "${CKPT}" ] || die "set LTX_CHECKPOINT to the checkpoint path"
[ -n "${GEMMA}" ] || die "set LTX_TEXT_ENCODER to the Gemma directory"
case "${CKPT}" in /*) ;; *) die "LTX_CHECKPOINT must be an absolute path: ${CKPT}" ;; esac
case "${GEMMA}" in /*) ;; *) die "LTX_TEXT_ENCODER must be an absolute path: ${GEMMA}" ;; esac
[ -f "${CKPT}" ] || die "checkpoint file does not exist: ${CKPT}"
[ -d "${GEMMA}" ] || die "text encoder directory does not exist: ${GEMMA}"

case "${INPUT}" in
  delete_disfluency)
    WORKFLOW=delete_disfluency
    SRC="${SCRIPT_DIR}/delete_disfluency_retake_input.mp4"
    DEFAULT_START=2.9667
    DEFAULT_END=3.9333
    ;;
  add_word)
    WORKFLOW=add_word
    SRC="${SCRIPT_DIR}/add_word_retake_input.mp4"
    DEFAULT_START=3.003
    DEFAULT_END=4.5045
    ;;
  replace_word)
    WORKFLOW=replace_word
    SRC="${SCRIPT_DIR}/replace_word_retake_input.mp4"
    DEFAULT_START=1.2679333333333334
    DEFAULT_END=2.5025
    ;;
  *)
    WORKFLOW=custom
    SRC="${INPUT}"
    DEFAULT_START=""
    DEFAULT_END=""
    ;;
esac
OUT_DIR="${OUT_DIR:-/tmp/ltx_retake/${WORKFLOW}}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
START="${LTX_START:-${DEFAULT_START}}"
END="${LTX_END:-${DEFAULT_END}}"
[ -n "${START}" ] && [ -n "${END}" ] || {
  echo "ERROR: set LTX_START and LTX_END when passing a custom source video" >&2
  exit 2
}
[ -f "${SRC}" ] || die "retake input does not exist: ${SRC}"
case "${SRC}" in
  /*) ;;
  *) SRC="$(cd "$(dirname "${SRC}")" && pwd)/$(basename "${SRC}")" ;;
esac

EX="$TRTLLM_REPO/examples/visual_gen/ltx2_retake_e2e.py"
[ -f "${EX}" ] || die "native retake example does not exist: ${EX}"

# Fail before model loading if the matching setup script was not run. Tag the
# capability output because importing TensorRT-LLM may also print a banner.
RUNTIME_INFO="$(cd /tmp && python3 - <<'PY'
import av  # noqa: F401
import torch
import torchaudio  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_retake import (
    LTX2RetakePipeline,  # noqa: F401
)

major, minor = torch.cuda.get_device_capability()
print(f"LTX_GPU_CC={major}.{minor}")
PY
)" || die "native retake runtime is not ready; run setup_hopper_env.sh or setup_sm120_env.sh first"
GPU_CC="$(printf '%s\n' "${RUNTIME_INFO}" | sed -n 's/^LTX_GPU_CC=//p' | tail -1)"
[ -n "${GPU_CC}" ] || die "could not determine the GPU compute capability"

for name in "${RECIPES[@]}"; do
  case "${name}" in
    bf16) ;;
    fp8|nvfp4)
      [ "${GPU_CC}" = 12.0 ] || die "recipe '${name}' requires the SM120 setup; detected compute capability ${GPU_CC}"
      ;;
    *) die "unsupported recipe '${name}' (expected bf16, fp8, or nvfp4)" ;;
  esac
done

# Neutral CWD so `import tensorrt_llm` resolves to the INSTALLED (overlaid)
# package instead of the uncompiled source tree under $TRTLLM_REPO.
cd /tmp
for name in "${RECIPES[@]}"; do
  case "${name}" in
    bf16) args=() ;;
    fp8) args=(--quant-algo FP8) ;;
    nvfp4) args=(--quant-algo NVFP4 --nvfp4-attn --fp8-linear-step 4 --fp8-linear-step 7) ;;
    *) die "unsupported recipe '${name}' (expected bf16, fp8, or nvfp4)" ;;
  esac
  echo "======== ${INPUT} / ${name} start $(date +%H:%M:%S) ========"
  python3 "$EX" \
    --checkpoint "$CKPT" --text-encoder "$GEMMA" \
    --source "$SRC" --output "${OUT_DIR}/retake_${name}.mp4" \
    --start "$START" --end "$END" --prompt "$PROMPT" "${args[@]}"
  echo "output: ${OUT_DIR}/retake_${name}.mp4"
  echo "======== ${INPUT} / ${name} end $(date +%H:%M:%S) ========"
done
