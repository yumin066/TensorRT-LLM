#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -eq 0 ]; then
  WORKFLOWS=(delete_disfluency add_word replace_word)
else
  WORKFLOWS=("$@")
fi

for workflow in "${WORKFLOWS[@]}"; do
  case "${workflow}" in
    delete_disfluency|add_word|replace_word) ;;
    *)
      echo "ERROR: unsupported workflow '${workflow}'; expected delete_disfluency, add_word, or replace_word" >&2
      exit 2
      ;;
  esac
done

# Compatibility wrapper for the tuned SM120 recipe. Model validation, prepared
# input selection, and timestamp selection are centralized in run_recipes.sh.
# With multiple workflows, an explicit OUT_DIR is treated as an output root so
# the cases cannot overwrite each other; for a single workflow it is preserved
# as the exact output directory for backward compatibility.
for workflow in "${WORKFLOWS[@]}"; do
  if [ -n "${OUT_DIR:-}" ] && [ "${#WORKFLOWS[@]}" -eq 1 ]; then
    workflow_out="${OUT_DIR}"
  else
    workflow_out="${OUT_DIR:-/tmp/ltx_retake}/${workflow}"
  fi
  mkdir -p "${workflow_out}"
  OUT_DIR="${workflow_out}" bash "${SCRIPT_DIR}/run_recipes.sh" "${workflow}" nvfp4 \
    2>&1 | tee "${workflow_out}/nvfp4.log"
done
