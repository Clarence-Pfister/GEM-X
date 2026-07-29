<h1 align="center">GEM-X — humanoid retargeting fork</h1>

<p align="center">
  <em>Fork of <a href="https://github.com/NVlabs/GEM-X">NVlabs/GEM-X</a>. This branch tracks upstream; the work lives on the branches below.</em>
</p>

---

## Branches

| Branch | Contents |
|---|---|
| `main` | Upstream NVlabs/GEM-X, plus this README. No functional changes. |
| `feature/g1-23dof-retargeting` | Adds a 23-DoF Unitree G1 retargeting target (`--retarget g1_23dof`). |
| `feature/ground-plane-estimation` | Ground-plane estimation to reduce foot-float / root drift in recovered motion. |
| `fix/onnxruntime-linux` | Installs `onnxruntime` on Linux, which upstream `install_env.sh` only does on macOS. Without it every demo fails at preprocess. |
| `fix/stale-requirements` | Removes `requirements.txt`, which nothing references and whose pins conflict with the documented install path. |
| `integration/all` | All of the above merged; the branch to actually run. |

Each feature branch is cut from `main` and is meant to stay independently
reviewable, so it can be proposed upstream on its own.

For model details, training, docs and license, see the official repository:
**https://github.com/NVlabs/GEM-X**

## Setup

```bash
git clone --recursive https://github.com/Clarence-Pfister/GEM-X.git && cd GEM-X
git checkout integration/all && git submodule update --init --recursive

pip install uv && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/soma && cd third_party/soma && git lfs pull && cd ../..
bash scripts/install_env.sh
uv pip install -e third_party/soma-retargeter
```

See [docs/INSTALL.md](docs/INSTALL.md) for the detailed upstream instructions.

## Retargeting

On `main` only the upstream 29-DoF target exists. The 23-DoF target requires
`feature/g1-23dof-retargeting` or `integration/all`.

```bash
source .venv/bin/activate

# 29-DoF G1 (upstream behaviour)
python scripts/demo/demo_soma.py --video path/to/video.mp4 --retarget

# 23-DoF G1
python scripts/demo/demo_soma.py --video path/to/video.mp4 --retarget g1_23dof
```

Add `--static_cam` when the video comes from a fixed camera.
The ONNX/TensorRT demo takes the same flag:

```bash
python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4 --retarget g1_23dof
```

### Outputs

Everything lands in `outputs/demo_soma/<video_name>/`:

| Robot | CSV | Video |
|---|---|---|
| 29-DoF | `<video_name>_retarget_g1.csv` | `<video_name>_4_g1_retarget.mp4` |
| 23-DoF | `<video_name>_retarget_g1_23dof.csv` | `<video_name>_4_g1_23dof_retarget.mp4` |

A `_from_bvh` CSV is also written for each run (BVH round-trip, for comparison).

### Input video

The pipeline is happiest with constant-framerate 720p input:

```bash
ffmpeg -i in.mp4 -vf "scale=1280:720" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a copy out.mp4
```

## License

Apache 2.0, same as upstream — see [LICENSE](LICENSE) and
[ATTRIBUTIONS.md](ATTRIBUTIONS.md).
Use of the associated model is governed by the
[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
