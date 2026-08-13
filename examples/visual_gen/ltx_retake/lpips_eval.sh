#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LPIPS-score delete-disfluency outputs on the fixed bridge window [89,118).
#   upstream golden <-> native bf16 (native implementation quality)
#   native bf16 <-> fp8 / nvfp4    (isolated quant effect)
# Outputs are read from OUT_DIR (default /tmp/ltx_retake/delete_disfluency).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
OUT="${OUT_DIR:-/tmp/ltx_retake/delete_disfluency}"
EVAL="$TRTLLM_REPO/scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
GOLDEN_ROOT="$TRTLLM_REPO/tests/integration/defs/examples/visual_gen/golden/visual_gen_lpips"
GOLDEN_ZIP="$GOLDEN_ROOT/visual_gen_lpips_golden_media.zip"
GOLDEN_CACHE="${LTX_GOLDEN_CACHE:-/tmp/ltx_retake/golden_media}"
GOLDEN="$GOLDEN_CACHE/ltx2_retake_lpips_golden_video.mp4"
DS="${OUT}/lpips_dataset.json"
RESULTS="${LPIPS_OUTPUT:-${OUT}/lpips_results.json}"
mkdir -p "${OUT}" "${GOLDEN_CACHE}" "$(dirname "${RESULTS}")"
BF16="$OUT/retake_bf16.mp4"
[ -f "${BF16}" ] || {
  echo "ERROR: run delete_disfluency BF16 first; missing ${BF16}" >&2
  exit 2
}
[ -f "${EVAL}" ] || {
  echo "ERROR: LPIPS evaluator does not exist: ${EVAL}" >&2
  exit 2
}

if [ ! -f "${GOLDEN}" ]; then
  [ -f "${GOLDEN_ZIP}" ] || {
    echo "ERROR: tracked golden archive does not exist: ${GOLDEN_ZIP}" >&2
    exit 2
  }
  python3 - "${GOLDEN_ZIP}" "${GOLDEN}" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive, destination = map(pathlib.Path, sys.argv[1:])
destination.parent.mkdir(parents=True, exist_ok=True)
try:
    with zipfile.ZipFile(archive) as source:
        with source.open(destination.name) as input_file, destination.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
except (zipfile.BadZipFile, KeyError) as error:
    raise SystemExit(
        f"ERROR: cannot extract {destination.name} from {archive}; "
        "fetch the Git LFS object first"
    ) from error
PY
fi

python3 -c 'import lpips' 2>/dev/null || python3 -m pip install --user --no-input lpips >/dev/null
python3 -c 'import cv2'   2>/dev/null || python3 -m pip install --user --no-input opencv-python-headless >/dev/null

# Build the dataset from available delete-disfluency outputs. Score the upstream
# golden against BF16 first, then isolate the optional quantization differences.
rows=()
rows+=("{\"reference_video_path\": \"$GOLDEN\", \"generated_video_path\": \"$BF16\", \"frame_start\": 89, \"frame_stop\": 118}")
for r in fp8 nvfp4; do
  if [ -f "$BF16" ] && [ -f "$OUT/retake_$r.mp4" ]; then
    rows+=("{\"reference_video_path\": \"$BF16\", \"generated_video_path\": \"$OUT/retake_$r.mp4\", \"frame_start\": 89, \"frame_stop\": 118}")
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
