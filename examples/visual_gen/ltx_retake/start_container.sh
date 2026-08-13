#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# Start the LTX-2.3 retake base container (detached), pinned to the tested
# image digest. See ./README.md for the customer runbook.
#
# The frozen contract uses a fixed rc22 digest rather than a mutable tag.
#
# Detached (`sleep infinity`) so the container survives across many
# `docker exec` calls and long runs; the gosu entrypoint runs it as the host
# user so files copied to bind mounts remain owned by the host user.
#
# Env overrides:
#   LTX_CHECKPOINT       absolute path to the LTX-2.3 checkpoint (required)
#   LTX_TEXT_ENCODER     absolute path to the Gemma directory (required)
#   LTX_CONTAINER_NAME   container name           (default ltx_retake)
#   LTX_IMAGE            built delivery image tag (default ltx-retake:rc22)
#   LTX_BASE_IMAGE       base image ref           (default: the pinned digest)
#   LTX_MOUNTS           optional space-separated absolute host paths to mount
#   NV_GPU               NVIDIA_VISIBLE_DEVICES   (default all)
#   LTX_REPLACE_CONTAINER=1 to replace an existing same-name container
# =============================================================================
set -euo pipefail

# Tested base image digest (frozen contract). Do not replace with a tag.
DEFAULT_BASE_IMAGE_DIGEST="nvcr.io/nvidia/tensorrt-llm/release@sha256:36d6146dfb19084f7098852991b0da67565cba1e5a9bb5980b1c1a7edae3bf1b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# examples/visual_gen/ltx_retake -> repo root is three levels up.
TRTLLM_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NAME="${LTX_CONTAINER_NAME:-ltx_retake}"
IMAGE="${LTX_IMAGE:-ltx-retake:rc22}"
BASE_IMAGE="${LTX_BASE_IMAGE:-${DEFAULT_BASE_IMAGE_DIGEST}}"
CHECKPOINT="${LTX_CHECKPOINT:-}"
TEXT_ENCODER="${LTX_TEXT_ENCODER:-}"
LTX_MOUNTS="${LTX_MOUNTS:-}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[ -n "${CHECKPOINT}" ] || die "set LTX_CHECKPOINT to the absolute host path of the LTX-2.3 checkpoint"
[ -n "${TEXT_ENCODER}" ] || die "set LTX_TEXT_ENCODER to the absolute host path of the Gemma directory"
case "${CHECKPOINT}" in /*) ;; *) die "LTX_CHECKPOINT must be an absolute path: ${CHECKPOINT}" ;; esac
case "${TEXT_ENCODER}" in /*) ;; *) die "LTX_TEXT_ENCODER must be an absolute path: ${TEXT_ENCODER}" ;; esac
[ -f "${CHECKPOINT}" ] || die "checkpoint file does not exist: ${CHECKPOINT}"
[ -d "${TEXT_ENCODER}" ] || die "text encoder directory does not exist: ${TEXT_ENCODER}"

replace_existing=0
if docker container inspect "${NAME}" >/dev/null 2>&1; then
  if [ "${LTX_REPLACE_CONTAINER:-0}" != 1 ]; then
    die "container '${NAME}' already exists; reuse it, remove it manually, or set LTX_REPLACE_CONTAINER=1"
  fi
  replace_existing=1
fi

# Build the thin delivery layer (ffmpeg + gosu entrypoint) on the pinned digest.
# Build context is this dir, which holds the Dockerfile + entrypoint.sh.
docker build "${SCRIPT_DIR}" \
  --tag "${IMAGE}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}"

# Keep an existing working container until its replacement image builds.
if [ "${replace_existing}" = 1 ]; then
  docker rm -f "${NAME}" >/dev/null
fi

args=(
  -d --name "${NAME}"
  --runtime=nvidia --ipc=host
  --entrypoint /entrypoint.sh
  -e "NVIDIA_VISIBLE_DEVICES=${NV_GPU:-all}"
  -e "HOST_USER_UID=$(id -u)"
  -e "HOST_USER_GID=$(id -g)"
  -e "UNAME=$(id -un)"
  -e "HOME=/tmp/ltx_retake_home"
  -e "TRTLLM_REPO=${TRTLLM_REPO}"
  -e "LTX_CHECKPOINT=${CHECKPOINT}"
  -e "LTX_TEXT_ENCODER=${TEXT_ENCODER}"
  # Do not use an NFS checkout as the OCI working directory: root-squashed NFS
  # can reject the runtime's pre-entrypoint chdir. Scripts are invoked by their
  # absolute mounted paths instead.
  -w /tmp
)
# Always mount this checkout and both model locations, plus configured extras.
mount_seen=" "
add_mount() {
  local path="$1"
  case "${path}" in /*) ;; *) die "mount path must be absolute: ${path}" ;; esac
  [ -e "${path}" ] || die "mount path does not exist: ${path}"
  case "${mount_seen}" in *" ${path} "*) return;; esac
  mount_seen="${mount_seen}${path} "
  args+=( -v "${path}:${path}" )
}
add_mount "${TRTLLM_REPO}"
add_mount "$(dirname "${CHECKPOINT}")"
add_mount "${TEXT_ENCODER}"
if [ -n "${LTX_MOUNTS}" ]; then
  read -r -a extra_mounts <<< "${LTX_MOUNTS}"
  for path in "${extra_mounts[@]}"; do add_mount "${path}"; done
fi

# `sleep infinity` is the CMD the gosu entrypoint execs as the host user.
docker run "${args[@]}" "${IMAGE}" sleep infinity >/dev/null
echo "started detached container '${NAME}' as $(id -un) (base digest pinned)"
echo "  Hopper:    docker exec -u $(id -u):$(id -g) ${NAME} bash ${SCRIPT_DIR}/setup_hopper_env.sh"
echo "  Blackwell: docker exec -u $(id -u):$(id -g) ${NAME} bash ${SCRIPT_DIR}/setup_sm120_env.sh"
