#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the native LTX-2 retake e2e for one or more quant recipes, saving outputs
# to /tmp (root-squash-safe; docker cp them out afterwards).
#
# Usage: run_recipes.sh <source.mp4> [recipe ...]   (default: bf16 fp8 nvfp4)
#
# Env overrides: LTX_CHECKPOINT, LTX_TEXT_ENCODER, LTX_PIPELINES_ROOT,
#                LTX_PROMPT, LTX_START, LTX_END, OUT_DIR.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CKPT="${LTX_CHECKPOINT:-/home/scratch.ylichen_sw/LTX2.3-script-editing/models/ltx-2.3-22b-distilled.safetensors}"
GEMMA="${LTX_TEXT_ENCODER:-/home/scratch.ylichen_sw/LTX2.3-script-editing/models/gemma}"
# Customer repo providing ltx_pipelines.media_io (source-video reader).
CUST="${LTX_PIPELINES_ROOT:-/home/scratch.minyu_gpu/project/shopee/LTX2.3-script-editing}"
PROMPT="${LTX_PROMPT:-a person talking to the camera, natural head motion, clear speech}"
START="${LTX_START:-3.0}"; END="${LTX_END:-3.9667}"
OUT_DIR="${OUT_DIR:-/tmp}"
SRC="${1:?usage: run_recipes.sh <source.mp4> [recipe ...]}"; shift || true
RECIPES=("$@"); [ ${#RECIPES[@]} -eq 0 ] && RECIPES=(bf16 fp8 nvfp4)

# Import-only stub for OpenImageIO (EXR image writer) which ltx_core imports at
# module load but the retake never calls. torchaudio is REAL (setup installs it):
# the retake audio conditioning encodes the source audio via MelSpectrogram, so
# it must NOT be shadowed by a stub.
STUB=/tmp/native_import_stubs
mkdir -p "$STUB"
rm -f "$STUB/torchaudio.py"
cat > "$STUB/OpenImageIO.py" <<'PY'
def __getattr__(n): raise NotImplementedError("stub OpenImageIO."+n)
PY
export PYTHONPATH="${STUB}:${CUST}/packages/ltx-pipelines/src:${CUST}/packages/ltx-core/src${PYTHONPATH:+:$PYTHONPATH}"
EX="$TRTLLM_REPO/examples/visual_gen/ltx2_retake_e2e.py"
# Neutral CWD so `import tensorrt_llm` resolves to the INSTALLED (overlaid)
# package instead of the uncompiled source tree under $TRTLLM_REPO.
cd /tmp
for name in "${RECIPES[@]}"; do
  args=()
  [ "$name" = "fp8" ]   && args=(--quant-algo FP8)
  [ "$name" = "nvfp4" ] && args=(--quant-algo NVFP4)
  echo "======== RECIPE ${name} start $(date +%H:%M:%S) ========"
  python3 "$EX" \
    --checkpoint "$CKPT" --text-encoder "$GEMMA" \
    --source "$SRC" --output "${OUT_DIR}/retake_${name}.mp4" \
    --start "$START" --end "$END" --prompt "$PROMPT" "${args[@]}"
  echo "======== RECIPE ${name} rc=$? end $(date +%H:%M:%S) ========"
done
