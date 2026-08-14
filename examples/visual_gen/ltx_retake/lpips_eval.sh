#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LPIPS-score one prepared workflow on its regenerated bridge window.
#   upstream golden <-> native bf16 (native implementation quality)
#   native bf16 <-> fp8 / nvfp4    (isolated quant effect)
# Usage: lpips_eval.sh [delete_disfluency|add_word|replace_word]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKFLOW="${1:-delete_disfluency}"
GOLDEN_ROOT="${SCRIPT_DIR}/golden_reference"
case "${WORKFLOW}" in
  delete_disfluency)
    GOLDEN="${GOLDEN_ROOT}/delete_disfluency_retake_output.mp4"
    FRAME_START=89
    FRAME_STOP=118
    ;;
  add_word)
    GOLDEN="${GOLDEN_ROOT}/add_word_retake_output.mp4"
    FRAME_START=90
    FRAME_STOP=135
    ;;
  replace_word)
    GOLDEN="${GOLDEN_ROOT}/replace_word_retake_output.mp4"
    FRAME_START=38
    FRAME_STOP=75
    ;;
  *)
    echo "ERROR: unsupported workflow '${WORKFLOW}'" >&2
    exit 2
    ;;
esac
OUT="${OUT_DIR:-/tmp/ltx_retake/${WORKFLOW}}"
EVAL="$TRTLLM_REPO/scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
DS="${OUT}/lpips_dataset.json"
RESULTS="${LPIPS_OUTPUT:-${OUT}/lpips_results.json}"
mkdir -p "${OUT}" "$(dirname "${RESULTS}")"
BF16="$OUT/retake_bf16.mp4"
[ -f "${BF16}" ] || {
  echo "ERROR: run ${WORKFLOW} BF16 first; missing ${BF16}" >&2
  exit 2
}
[ -f "${EVAL}" ] || {
  echo "ERROR: LPIPS evaluator does not exist: ${EVAL}" >&2
  exit 2
}
[ -f "${GOLDEN}" ] || {
  echo "ERROR: upstream golden is missing: ${GOLDEN}; run git lfs pull" >&2
  exit 2
}

python3 -c 'import lpips' 2>/dev/null || python3 -m pip install --user --no-input lpips >/dev/null
python3 -c 'import cv2'   2>/dev/null || python3 -m pip install --user --no-input opencv-python-headless >/dev/null

# Score the upstream golden against BF16 first, then isolate optional
# quantization differences over the same regenerated bridge window.
rows=()
rows+=("{\"reference_video_path\": \"$GOLDEN\", \"generated_video_path\": \"$BF16\", \"frame_start\": $FRAME_START, \"frame_stop\": $FRAME_STOP}")
for r in fp8 nvfp4; do
  if [ -f "$BF16" ] && [ -f "$OUT/retake_$r.mp4" ]; then
    rows+=("{\"reference_video_path\": \"$BF16\", \"generated_video_path\": \"$OUT/retake_$r.mp4\", \"frame_start\": $FRAME_START, \"frame_stop\": $FRAME_STOP}")
  fi
done
if [ ${#rows[@]} -eq 0 ]; then
  echo "ERROR: no comparable retake outputs found in ${OUT}" >&2
  exit 1
fi
{ echo '['; ( IFS=$'\n'; printf '  %s,\n' "${rows[@]}" | sed '$ s/,$//' ); echo ']'; } > "$DS"

cd /tmp
python3 "$EVAL" --dataset "$DS" --lpips-net alex --output-json "$RESULTS"
echo "=== ${RESULTS} ==="
cat "$RESULTS"
