# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# The CUDA wheels pulled below are large (cudnn 674MB, cublas 375MB,
# cusparselt 274MB). uv's default 30s HTTP timeout is easily exceeded on
# an ordinary connection, failing the build mid-download.
ENV UV_HTTP_TIMEOUT=600

# System dependencies.
# Ubuntu 22.04 only ships Python 3.10; it is installed here solely to
# bootstrap pip/uv. The project venv itself is created on 3.12 below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    gosu \
    xvfb \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Install uv
RUN pip install uv

# Set up workspace
WORKDIR /workspace/gem
COPY . /workspace/gem

# Create virtual environment.
# Python 3.12 is required: docs/INSTALL.md states 3.12+, and
# third_party/soma-retargeter declares requires-python = ">=3.12", so a 3.10
# venv makes the retargeting step in INSTALL.md Step 6 uninstallable.
# uv fetches a standalone 3.12 build, so no deadsnakes PPA is needed.
RUN uv python install 3.12
RUN uv venv .venv --python 3.12

# Install PyTorch
RUN . .venv/bin/activate && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install SOMA body model
RUN . .venv/bin/activate && \
    uv pip install -e third_party/soma

# Pull SOMA LFS assets
RUN cd third_party/soma && git lfs pull && cd ../..

# Install GEM and dependencies
RUN . .venv/bin/activate && \
    bash scripts/install_env.sh

# Headless rendering environment
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless

# Activate venv by default
ENV PATH="/workspace/gem/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/workspace/gem/.venv"

ENTRYPOINT ["tools/docker-entrypoint.sh"]
CMD ["bash"]
