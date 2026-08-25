# Installation

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU with drivers supporting CUDA 12.6+
- [Git LFS](https://git-lfs.github.com/) (required for SOMA body model assets)
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

## Step 1 — Clone with submodules

```bash
git clone --recursive https://github.com/NVlabs/GEM-X.git
cd GEM-X
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

## Step 2 — Create virtual environment

```bash
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
```

## Step 3 — Install PyTorch with CUDA

```bash
# Adjust the CUDA version to match your GPU driver.
# See https://pytorch.org/get-started/locally/
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

| CUDA Version | Index URL |
|---|---|
| CUDA 12.6 | `https://download.pytorch.org/whl/cu126` |
| CUDA 13.0 | `https://download.pytorch.org/whl/cu130` |

## Step 4 — Install SOMA body model

```bash
uv pip install -e third_party/soma
cd third_party/soma && git lfs pull && cd ../..
```

## Step 5 — Install GEM and dependencies

```bash
bash scripts/install_env.sh
```

This installs the `gem` package in editable mode along with Detectron2 for human detection.

## Step 6 — (Optional) Install SOMA Retargeter for humanoid robot retargeting

To enable `--retarget` mode (retarget recovered motion to the Unitree G1 robot):

```bash
uv pip install -e third_party/soma-retargeter
```

## Step 7 — Third-party model assets

**SOMA body model** — follow `third_party/soma/README.md` and place model assets under `inputs/soma_assets/`.

**SAM-3D-Body** — follow `third_party/sam-3d-body/README.md` to download the checkpoint.

## Pretrained Model Download

Download the pretrained GEM checkpoint:

- **GEM (SOMA)**: [gem_soma.ckpt](https://huggingface.co/nvidia/GEM-X)

You can also download manually via CLI:
```bash
huggingface-cli download nvidia/GEM-X gem_soma.ckpt --local-dir inputs/pretrained
```

Place it under `inputs/pretrained/` or pass the path via `--ckpt`.

## Expected Directory Layout

After setup, your `inputs/` directory should look like:

```
inputs/
├── pretrained/
│   └── gem_soma.ckpt
├── soma_assets/
│   ├── soma_model/
│   └── ...
└── sam3d/
    └── checkpoint.pth
```

## Docker

A `Dockerfile` is provided at the repository root for reproducible setup. See the [Dockerfile](../Dockerfile) for details.

## Troubleshooting

| Issue | Solution |
|---|---|
| `git lfs` files are pointer files | Run `cd third_party/soma && git lfs pull` |
| CUDA version mismatch | Ensure PyTorch CUDA version matches your driver (`nvidia-smi`) |
| `ModuleNotFoundError: gem` | Ensure you ran `bash scripts/install_env.sh` with the venv activated |
| OpenGL/EGL errors | Set `PYOPENGL_PLATFORM=egl` and `EGL_PLATFORM=surfaceless` |
