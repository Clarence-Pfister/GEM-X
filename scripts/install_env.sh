#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "Installing gem in editable mode..."
uv pip install -e .

echo "Installing SAM-3D-Body runtime deps..."
uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'

# ONNX Runtime is required on every platform: the bundled YOLOX detector
# (gem/utils/yolox_detector.py) imports it for human detection in all demos.
if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected — skipping detectron2, installing ONNX Runtime..."
    uv pip install onnxruntime
else
    echo "Installing detectron2..."
    uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
    # onnxruntime-gpu 1.27+ links CUDA 13 (libcudart.so.13). This project
    # targets CUDA 12.6, so pin below that or the import fails at runtime.
    echo "Installing ONNX Runtime (CUDA 12 GPU build for CUDAExecutionProvider)..."
    uv pip install 'onnxruntime-gpu<1.27'
fi

echo "Environment setup complete."
