#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Build the LTX-2.3 retake SM120 environment inside the pinned rc22 container.
# NATIVE (integrated) variant: the whole retake workflow runs in the container's
# SYSTEM python via the native TensorRT-LLM VisualGen pipeline, so there is no
# separate Step A / Step C and this does NOT run `uv sync` / build a `.venv`.
#
# RUN THIS INSIDE the container started by ./start_sm120_container.sh.
#
# Idempotent, all against the SYSTEM python:
#   1. overlay the LTX python from THIS TensorRT-LLM checkout onto the rc22
#      pre-compiled install: the whole _torch/visual_gen subtree plus
#      _torch/modules/linear.py (SM120 NVFP4). Only these LTX files are
#      overlaid -- the rest of the repo python is newer than rc22 and would fail
#      against rc22's .so. No recompile.
#   2. build + install flashinfer-python 0.6.17 from FlashInfer PR #4272
#      (SM120 nvfp4 attention kernels; stock PyPI flashinfer lacks them)
#   3. native-retake runtime deps: PyAV, plus import stubs for the audio/image
#      libs (torchaudio / OpenImageIO) that ltx_pipelines.media_io imports at
#      module load but the video-only retake never calls
#   4. environment preflight self-check
# =============================================================================
set -euo pipefail

# ---- Frozen versions (do not bump without re-freezing the whole contract) ----
# Anchor SHA of jaywan's SM120 base (this checkout carries it plus the retake
# changes); used only for the preflight sanity note.
TRTLLM_OVERLAY_BASE_SHA="c6e90fbc23db35ec41cf7bd3757d124004137412"

FLASHINFER_PR_SHA="2e50aa4af56f793eb9190bc1912786df6b5fd038"
FLASHINFER_REPO="https://github.com/flashinfer-ai/flashinfer.git"
# Deliverable default: fetch PR #4272 straight from GitHub (customers cannot
# reach our cluster). FLASHINFER_SHARED_ARCHIVE is an OPTIONAL internal-only
# override -- point it at an extracted-source tarball for air-gapped hosts.
FLASHINFER_SHARED_ARCHIVE="${FLASHINFER_SHARED_ARCHIVE:-}"

# ---- Paths (self-rooted at this checkout) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# examples/visual_gen/ltx2_retake_sm120 -> repo root is three levels up.
TRTLLM_REPO="${TRTLLM_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DEPS_DIR="${TRTLLM_REPO}/.deps"
STUB_DIR="${DEPS_DIR}/native_import_stubs"

log() { printf '\n== %s ==\n' "$*"; }

# ---- 1. LTX python overlay (this checkout -> rc22 pre-compiled install) ------
log "1/4 LTX overlay from ${TRTLLM_REPO}"
VG_SRC="${TRTLLM_REPO}/tensorrt_llm/_torch/visual_gen"
LINEAR_SRC="${TRTLLM_REPO}/tensorrt_llm/_torch/modules/linear.py"
[ -f "${LINEAR_SRC}" ] || {
  echo "ERROR: TensorRT-LLM checkout not found at ${TRTLLM_REPO} (is it mounted into the container? set TRTLLM_REPO=...)" >&2
  exit 1
}
grep -q apply_with_pre_quant_scale_applied "${LINEAR_SRC}" || {
  echo "ERROR: linear.py lacks the pre-smoothed NVFP4 API (wrong/old checkout)" >&2
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
sudo cp "${LINEAR_SRC}" "${TRTLLM_PKG}/_torch/modules/linear.py"
test -f "${TRTLLM_PKG}/_torch/visual_gen/models/ltx2/pipeline_ltx2_retake.py"
test -f "${TRTLLM_PKG}/_torch/visual_gen/models/ltx2/retake_adapter.py"
grep -q apply_with_pre_quant_scale_applied "${TRTLLM_PKG}/_torch/modules/linear.py"
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
log "3/4 native-retake deps: PyAV + audio-lib import stubs"
python3 -m pip install --no-input av
mkdir -p "${STUB_DIR}"
cat > "${STUB_DIR}/torchaudio.py" <<'PYSTUB'
"""Import-only stub: video-only retake never calls torchaudio ops."""
class _Raiser:
    def __getattr__(self, name):
        raise NotImplementedError(f"stub torchaudio.{name}: audio ops unavailable")
transforms = _Raiser()
functional = _Raiser()
__version__ = "0.0.0+stub"
PYSTUB
cat > "${STUB_DIR}/OpenImageIO.py" <<'PYSTUB'
"""Import-only stub: video-only retake never touches the EXR image writer."""
def __getattr__(name):
    raise NotImplementedError(f"stub OpenImageIO.{name}: image I/O unavailable")
PYSTUB

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
from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod

assert torch.version.cuda == "13.2", torch.version.cuda
assert torch.cuda.get_device_capability() == (12, 0), torch.cuda.get_device_capability()
assert flashinfer.__version__ == "0.6.17", flashinfer.__version__
assert hasattr(flashinfer, "nvfp4_attention_sm120_quantize_qkv")
assert hasattr(flashinfer, "nvfp4_attention_sm120_fwd")
assert callable(fmha_v2_prefill_sm120)
assert hasattr(NVFP4LinearMethod, "apply_with_pre_quant_scale_applied")
assert hasattr(NVFP4LinearMethod, "configure_dynamic_absmax_backend")
assert hasattr(NVFP4LinearMethod, "dynamic_absmax_report")
import av  # noqa: F401  native source/output video I/O
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flashinfer:", flashinfer.__version__, pathlib.Path(flashinfer.__file__).resolve())
print("tensorrt_llm:", tensorrt_llm.__version__)
print("gpu:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("environment preflight: PASS")
PY

log "SM120 native-retake environment ready"
cat <<EOF

Import stubs live at: ${STUB_DIR}
At RUN time the retake runner (run_recipes.sh) prepends the stubs and the
customer ltx-pipelines/ltx-core src to PYTHONPATH (media_io source reader).
EOF
