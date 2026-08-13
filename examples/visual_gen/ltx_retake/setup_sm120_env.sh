#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Build the LTX-2.3 retake SM120 environment inside the pinned rc22 container.
# The whole workflow runs in the container's system Python through the native
# TensorRT-LLM VisualGen pipeline; no separate virtual environment is needed.
#
# RUN THIS INSIDE the container started by ./start_container.sh.
#
# Idempotent, all against the SYSTEM python:
#   1. overlay the VisualGen Python tree and its Linear dependency from this
#      checkout onto the rc22 pre-compiled install.
#   2. build + install flashinfer-python 0.6.17 from FlashInfer PR #4272
#      (SM120 nvfp4 attention kernels; stock PyPI flashinfer lacks them)
#   3. native-retake runtime deps: PyAV + torchaudio
#   4. environment preflight self-check
# =============================================================================
set -euo pipefail

FLASHINFER_PR_SHA="2e50aa4af56f793eb9190bc1912786df6b5fd038"
FLASHINFER_REPO="https://github.com/flashinfer-ai/flashinfer.git"
# Fetch PR #4272 straight from GitHub by default. For an air-gapped setup,
# FLASHINFER_SHARED_ARCHIVE may point to a mounted extracted-source tarball.
FLASHINFER_SHARED_ARCHIVE="${FLASHINFER_SHARED_ARCHIVE:-}"

# ---- Paths (self-rooted at this checkout) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# examples/visual_gen/ltx_retake -> repo root is three levels up.
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DEPS_DIR="${TRTLLM_REPO}/.deps"

log() { printf '\n== %s ==\n' "$*"; }

# ---- 1. LTX python overlay (this checkout -> rc22 pre-compiled install) ------
log "1/4 LTX overlay from ${TRTLLM_REPO}"
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

# ---- 2. FlashInfer 0.6.17 from PR #4272 (SM120 nvfp4 attention) --------------
log "2/4 flashinfer-python 0.6.17 (PR #4272 @ ${FLASHINFER_PR_SHA})"
mkdir -p "${DEPS_DIR}/wheels"
# The build (~8 min) is the slow part; the wheel lives on scratch, so reuse it
# across containers. This keeps a re-run fast enough to finish inside the ~4 min
# container-reap window on shared "gen" nodes.
FI_WHEEL="$(ls "${DEPS_DIR}"/wheels/flashinfer_python-0.6.17-*.whl 2>/dev/null | head -1 || true)"
if [ -z "${FI_WHEEL}" ]; then
  # Build in a LOCAL dir, not on NFS: compiling with build intermediates on NFS
  # is several times slower. The resulting wheel still lands in DEPS_DIR/wheels
  # (persistent, reused across containers).
  FI_SRC="${FI_BUILD_DIR:-/tmp/flashinfer-pr4272-build}"
  if [ ! -d "${FI_SRC}" ]; then
    if [ -n "${FLASHINFER_SHARED_ARCHIVE}" ] && [ -f "${FLASHINFER_SHARED_ARCHIVE}" ]; then
      echo "  internal override: extracting FLASHINFER_SHARED_ARCHIVE"
      mkdir -p "${FI_SRC}"
      tar -xzf "${FLASHINFER_SHARED_ARCHIVE}" -C "${FI_SRC}"
    else
      echo "  fetching FlashInfer PR #4272 commit from GitHub (by SHA)"
      git init -q "${FI_SRC}"
      git -C "${FI_SRC}" remote add origin "${FLASHINFER_REPO}"
      # GitHub serves the exact commit even after the PR head advances.
      git -C "${FI_SRC}" fetch --depth 1 origin "${FLASHINFER_PR_SHA}"
      git -C "${FI_SRC}" checkout -q --detach FETCH_HEAD
      git -C "${FI_SRC}" submodule update --init --recursive --depth 1
      test "$(git -C "${FI_SRC}" rev-parse HEAD)" = "${FLASHINFER_PR_SHA}"
    fi
  fi
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export FLASHINFER_CUDA_ARCH_LIST=12.0a
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export BUILD_NIXL_EP=0
  export BUILD_NCCL_EP=0
  export MAX_JOBS="${MAX_JOBS:-8}"
  python3 -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "${DEPS_DIR}/wheels" "${FI_SRC}"
  FI_WHEEL="$(ls "${DEPS_DIR}"/wheels/flashinfer_python-0.6.17-*.whl | head -1)"
else
  echo "  reusing cached wheel: ${FI_WHEEL}"
fi
sudo -E python3 -m pip install --no-deps --force-reinstall "${FI_WHEEL}"

# ---- 3. Native-retake runtime deps (system python) --------------------------
log "3/4 native-retake deps: PyAV + torchaudio"
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
# ---- 4. Environment preflight ----------------------------------------------
log "4/4 environment preflight"
cd /tmp  # neutral CWD so `import tensorrt_llm` hits the INSTALLED package, not
         # the uncompiled source tree (docker exec starts in the repo root).
python3 - <<'PY'
import pathlib
import torch
import flashinfer
import tensorrt_llm
from flashinfer.prefill import fmha_v2_prefill_sm120
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.media_io import (
    decode_video_by_frame,
    get_videostream_metadata,
)

assert torch.version.cuda == "13.2", torch.version.cuda
assert torch.cuda.get_device_capability() == (12, 0), torch.cuda.get_device_capability()
assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
assert hasattr(flashinfer, "nvfp4_attention_sm120_quantize_qkv")
assert hasattr(flashinfer, "nvfp4_attention_sm120_fwd")
assert callable(fmha_v2_prefill_sm120)
import av  # noqa: F401  native source/output video I/O
import torchaudio  # noqa: F401  retake audio conditioning (mel/resample)
assert callable(decode_video_by_frame)
assert callable(get_videostream_metadata)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flashinfer:", flashinfer.__version__, pathlib.Path(flashinfer.__file__).resolve())
print("tensorrt_llm:", tensorrt_llm.__version__)
print("gpu:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("environment preflight: PASS")
PY

log "SM120 native-retake environment ready"
