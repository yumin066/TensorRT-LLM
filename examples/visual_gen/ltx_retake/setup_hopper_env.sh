#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Build the LTX-2.3 retake environment inside the pinned rc22 container on a
# HOPPER host (H100 / H200, SM90). This is the Hopper counterpart of
# ./setup_sm120_env.sh -- same overlay + native-retake runtime deps, but WITHOUT
# the SM120-only FlashInfer PR #4272 build (its nvfp4 attention kernels are
# compiled for arch 12.0a and neither build nor load on Hopper). The bf16 retake
# path uses the base image's attention backend (TRTLLM / SDPA), so no extra
# FlashInfer is needed for the BF16 recipe.
#
# RUN THIS INSIDE the container started by ./start_container.sh (the same
# pinned rc22 base image is used on Hopper and Blackwell).
#
# Idempotent, all against the SYSTEM python:
#   1. overlay the VisualGen Python tree and its Linear dependency from this
#      checkout onto the rc22 pre-compiled install.
#   2. native-retake runtime deps: PyAV + torchaudio
#   3. environment preflight self-check (Hopper / SM90)
#
# NOTE on non-BF16 recipes: the FP8 / NVFP4 attention fast paths are SM120-tuned
# (see setup_sm120_env.sh). On Hopper the validation target is BF16; FP8/NVFP4
# would need a Hopper-tuned FlashInfer and are out of scope for this script.
# =============================================================================
set -euo pipefail

# ---- Paths (self-rooted at this checkout) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# examples/visual_gen/ltx_retake -> repo root is three levels up.
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
log() { printf '\n== %s ==\n' "$*"; }

# ---- 1. LTX python overlay (this checkout -> rc22 pre-compiled install) ------
log "1/3 LTX overlay from ${TRTLLM_REPO}"
VG_SRC="${TRTLLM_REPO}/tensorrt_llm/_torch/visual_gen"
LINEAR_SRC="${TRTLLM_REPO}/tensorrt_llm/_torch/modules/linear.py"
[ -d "${VG_SRC}" ] && [ -f "${LINEAR_SRC}" ] || {
  echo "ERROR: TensorRT-LLM checkout not found at ${TRTLLM_REPO} (is it mounted into the container? set TRTLLM_REPO=...)" >&2
  exit 1
}
# Locate the pre-compiled tensorrt_llm inside the container (do not hardcode the
# python version / dist-packages path). Importing tensorrt_llm prints a version
# banner to stdout, so tag the real answer with a marker and extract only that.
# Run from a neutral CWD so it does not import the (uncompiled) source tree.
TRTLLM_PKG="$(cd /tmp && python3 -c 'import os,tensorrt_llm; print("TRTLLM_PKG="+os.path.dirname(tensorrt_llm.__file__))' 2>/dev/null | sed -n 's/^TRTLLM_PKG=//p')"
echo "  installed tensorrt_llm: ${TRTLLM_PKG}"
# Overlay the whole VisualGen subtree, python only, preserving dir structure and
# leaving rc22's compiled extensions in place.
( cd "${VG_SRC}" && find . -name '*.py' -print0 | tar --null -T - -cf - ) \
  | sudo tar -xf - -C "${TRTLLM_PKG}/_torch/visual_gen"
LINEAR_STAGE="$(mktemp /tmp/ltx2-linear.XXXXXX.py)"
trap 'rm -f "${LINEAR_STAGE}"' EXIT
cp "${LINEAR_SRC}" "${LINEAR_STAGE}"
sudo cp "${LINEAR_STAGE}" "${TRTLLM_PKG}/_torch/modules/linear.py"
test -f "${TRTLLM_PKG}/_torch/visual_gen/models/ltx2/pipeline_ltx2_retake.py"
echo "  overlaid: _torch/visual_gen/**.py + _torch/modules/linear.py"

# ---- 2. Native-retake runtime deps (system python) --------------------------
log "2/3 native-retake deps: PyAV + torchaudio"
python3 -m pip install --no-input av
# Real torchaudio: the retake audio conditioning encodes the source audio via
# torchaudio.transforms.MelSpectrogram + functional.resample (pure-torch ops).
# There is no torchaudio wheel matching this torch+CUDA build, so install the
# closest one (--no-deps, does not touch torch) and neuter its import-time CUDA
# version check -- the C++ codec extension it guards is unused by the mel/resample
# transforms, which are pure PyTorch.
python3 -m pip install --no-deps --no-input "torchaudio==2.11.0"
_TA_INIT="$(python3 -c 'import importlib.util,os; s=importlib.util.find_spec("torchaudio"); print(os.path.join(os.path.dirname(s.origin),"_extension","__init__.py"))' 2>/dev/null || true)"
if [ -n "${_TA_INIT}" ] && [ -f "${_TA_INIT}" ]; then
  sudo sed -i 's/^\(\s*\)_check_cuda_version()/\1pass  # patched: torchaudio\/torch CUDA mismatch; mel\/resample are pure-torch/' "${_TA_INIT}"
fi
# ---- 3. Environment preflight (Hopper / SM90) -------------------------------
log "3/3 environment preflight (Hopper)"
cd /tmp  # neutral CWD so `import tensorrt_llm` hits the INSTALLED package, not
         # the uncompiled source tree (docker exec starts in the repo root).
python3 - <<'PY'
import torch
import tensorrt_llm
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.media_io import (
    decode_video_by_frame,
    get_videostream_metadata,
)

cc = torch.cuda.get_device_capability()
# Hopper is SM90 (H100 / H200). Reject a Blackwell/SM120 host: that needs the
# SM120 FlashInfer build from setup_sm120_env.sh, not this script.
assert cc == (9, 0), f"expected Hopper SM90 (9, 0), got {cc}"
import av  # noqa: F401  native source/output video I/O
import torchaudio  # noqa: F401  retake audio conditioning (mel/resample)
assert callable(decode_video_by_frame)
assert callable(get_videostream_metadata)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("tensorrt_llm:", tensorrt_llm.__version__)
print("gpu:", torch.cuda.get_device_name(), cc)
print("environment preflight: PASS")
PY

log "Hopper native-retake environment ready (bf16)"
