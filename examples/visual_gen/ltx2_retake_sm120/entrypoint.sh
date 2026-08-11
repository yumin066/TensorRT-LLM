#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Create the host user inside the container and drop privileges to it, so files
# written to the user's NFS scratch stay owned by them (not root-squashed).
set -e

USERNAME="${UNAME:?UNAME env var is required}"
USER_UID="${HOST_USER_UID:-1000}"
USER_GID="${HOST_USER_GID:-1000}"

if ! getent group "$USER_GID" > /dev/null; then
    groupadd -g "$USER_GID" "$USERNAME"
fi

if ! getent passwd "$USER_UID" > /dev/null; then
    useradd -l -u "$USER_UID" -g "$USER_GID" -m "$USERNAME"
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

exec gosu "$USERNAME" "$@"
