#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LPIPS-score the retake outputs on the bridge window [90,119).
#   golden      <-> native bf16   (native-vs-frozen calibration; if golden present)
#   native bf16 <-> fp8 / nvfp4    (isolated quant effect)
# Outputs are read from OUT_DIR (default /tmp/retake_<recipe>.mp4).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
OUT="${OUT_DIR:-/tmp}"
EVAL="$TRTLLM_REPO/scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
GOLDEN="$TRTLLM_REPO/tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips/visual_gen_lpips_golden_media/ltx2_retake_lpips_golden_video.mp4"
DS=/tmp/lpips_dataset.json

python3 -c 'import lpips' 2>/dev/null || pip install --no-input lpips >/dev/null 2>&1
python3 -c 'import cv2'   2>/dev/null || pip install --no-input opencv-python-headless >/dev/null 2>&1

# Build the dataset from whatever outputs exist (skip missing recipes, e.g.
# nvfp4 on a non-Blackwell GPU). golden<->bf16 first if the golden is present.
rows=()
BF16="$OUT/retake_bf16.mp4"
if [ -f "$GOLDEN" ] && [ -f "$BF16" ]; then
  rows+=("{\"reference_video_path\": \"$GOLDEN\", \"generated_video_path\": \"$BF16\", \"frame_start\": 90, \"frame_stop\": 119}")
fi
for r in fp8 nvfp4; do
  if [ -f "$BF16" ] && [ -f "$OUT/retake_$r.mp4" ]; then
    rows+=("{\"reference_video_path\": \"$BF16\", \"generated_video_path\": \"$OUT/retake_$r.mp4\", \"frame_start\": 90, \"frame_stop\": 119}")
  fi
done
{ echo '['; ( IFS=$'\n'; printf '  %s,\n' "${rows[@]}" | sed '$ s/,$//' ); echo ']'; } > "$DS"

cd /tmp
python3 "$EVAL" --dataset "$DS" --lpips-net alex --output-json /tmp/lpips_results.json
echo "=== /tmp/lpips_results.json ==="
cat /tmp/lpips_results.json
