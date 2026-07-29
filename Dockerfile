# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# The CUDA wheels pulled below are large (cudnn 674MB, cublas 375MB,
# cusparselt 274MB). uv's default 30s HTTP timeout is easily exceeded on
# an ordinary connection, failing the build mid-download.
ENV UV_HTTP_TIMEOUT=600

# System dependencies
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

# Create virtual environment
RUN uv venv .venv --python 3.10

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
