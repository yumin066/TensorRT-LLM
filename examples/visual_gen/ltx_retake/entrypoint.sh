#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ensure the host UID/GID has an account inside the container, give it a writable
# HOME and passwordless sudo for the environment setup, then drop privileges.
set -euo pipefail

REQUESTED_USERNAME="${UNAME:?UNAME env var is required}"
USER_UID="${HOST_USER_UID:-1000}"
USER_GID="${HOST_USER_GID:-1000}"
USER_HOME="${HOME:-/tmp/ltx_retake_home}"

if ! getent group "${USER_GID}" >/dev/null; then
    GROUP_NAME="${REQUESTED_USERNAME}"
    if getent group "${GROUP_NAME}" >/dev/null; then
        GROUP_NAME="ltx_group_${USER_GID}"
    fi
    groupadd -g "${USER_GID}" "${GROUP_NAME}"
fi

RUN_USERNAME="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"
if [ -z "${RUN_USERNAME}" ]; then
    RUN_USERNAME="${REQUESTED_USERNAME}"
    if getent passwd "${RUN_USERNAME}" >/dev/null; then
        RUN_USERNAME="ltx_user_${USER_UID}"
    fi
    useradd -l -u "${USER_UID}" -g "${USER_GID}" -d "${USER_HOME}" -M "${RUN_USERNAME}"
fi

mkdir -p "${USER_HOME}"
chown "${USER_UID}:${USER_GID}" "${USER_HOME}"
if [ "${USER_UID}" != 0 ]; then
    echo "${RUN_USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/ltx-retake-${USER_UID}"
    chmod 0440 "/etc/sudoers.d/ltx-retake-${USER_UID}"
fi

export HOME="${USER_HOME}"
export USER="${RUN_USERNAME}"
export LOGNAME="${RUN_USERNAME}"
exec gosu "${USER_UID}:${USER_GID}" "$@"
