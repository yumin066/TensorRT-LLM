#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Start the LTX-2.3 retake SM120 base container (DETACHED), pinned to the tested
# image digest. See ./README.md for the full runbook.
#
# The frozen contract uses a FIXED rc22 digest (CUDA 13.2, tested against a
# 595.x SM120 driver), NOT a mutable tag. A mismatched image on Blackwell makes
# advanced CUDA ops fault ("unknown NVRM ioctl") and crashes the container, so
# the digest pin is load-bearing.
#
# Detached (`sleep infinity`) so the container survives across many
# `docker exec` calls and long runs; the gosu entrypoint runs it as the host
# user so writes to the user's NFS scratch stay owned by them (not root-squashed).
#
# Env overrides:
#   LTX_CONTAINER_NAME   container name           (default ltx_sm120)
#   LTX_IMAGE            built delivery image tag (default ltx23-sm120:rc22)
#   LTX_BASE_IMAGE       base image ref           (default: the pinned digest)
#   LTX_MOUNTS           space-separated host paths to bind-mount (default: the
#                        scratch roots holding the checkpoint / Gemma / customer
#                        ltx-pipelines / flashinfer archive)
#   NV_GPU               NVIDIA_VISIBLE_DEVICES   (default all)
# =============================================================================
set -euo pipefail

# Tested base image digest (frozen contract). Do not replace with a tag.
DEFAULT_BASE_IMAGE_DIGEST="nvcr.io/nvidia/tensorrt-llm/release@sha256:36d6146dfb19084f7098852991b0da67565cba1e5a9bb5980b1c1a7edae3bf1b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# examples/visual_gen/ltx2_retake_sm120 -> repo root is three levels up.
TRTLLM_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NAME="${LTX_CONTAINER_NAME:-ltx_sm120}"
IMAGE="${LTX_IMAGE:-ltx23-sm120:rc22}"
BASE_IMAGE="${LTX_BASE_IMAGE:-${DEFAULT_BASE_IMAGE_DIGEST}}"
# Default mounts cover: this TensorRT-LLM checkout + customer ltx-pipelines
# (scratch.minyu_gpu), the 22B checkpoint + Gemma (scratch.ylichen_sw), and the
# flashinfer PR#4272 source archive (scratch.jaywan_sw).
LTX_MOUNTS="${LTX_MOUNTS:-/home/scratch.minyu_gpu /home/scratch.ylichen_sw /home/scratch.jaywan_sw}"

# Build the thin delivery layer (ffmpeg + gosu entrypoint) on the pinned digest.
# Build context is this dir, which holds the Dockerfile + entrypoint.sh.
docker build "${SCRIPT_DIR}" \
  --tag "${IMAGE}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}"

docker rm -f "${NAME}" 2>/dev/null || true

args=(
  -d --name "${NAME}"
  --runtime=nvidia --ipc=host
  --entrypoint /entrypoint.sh
  -e "NVIDIA_VISIBLE_DEVICES=${NV_GPU:-all}"
  -e "HOST_USER_UID=$(id -u)"
  -e "HOST_USER_GID=$(id -g)"
  -e "UNAME=$(id -un)"
  -w "${TRTLLM_REPO}"
)
# Always mount this checkout (the overlay source), plus the configured extras.
mount_seen=" "
add_mount() {
  case "${mount_seen}" in *" $1 "*) return;; esac
  mount_seen="${mount_seen}$1 "
  args+=( -v "$1:$1" )
}
add_mount "${TRTLLM_REPO}"
for m in ${LTX_MOUNTS}; do add_mount "${m}"; done

# `sleep infinity` is the CMD the gosu entrypoint execs as the host user.
docker run "${args[@]}" "${IMAGE}" sleep infinity >/dev/null
echo "started detached container '${NAME}' as $(id -un) (base digest pinned)"
echo "  build env:  docker exec -u $(id -un) ${NAME} bash ${SCRIPT_DIR}/setup_sm120_env.sh"
