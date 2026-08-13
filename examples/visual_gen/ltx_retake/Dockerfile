# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Thin delivery layer over the pinned rc22 base: adds the tools the retake run
# needs (ffmpeg for source/output video, sudo/gosu for the drop-to-user
# entrypoint) without touching the pre-compiled TensorRT-LLM install.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    sudo \
    gosu \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
