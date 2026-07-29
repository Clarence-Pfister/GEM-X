# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground-plane estimation from visible-ground masks and monocular depth.

The estimator is intentionally optional: callers can provide precomputed depth /
mask arrays, or local HuggingFace/transformers model names. If no depth source is
available, it skips rather than pretending a 2D mask is a metric 3D floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gem.utils.pylogger import Log

GROUND_LABEL_KEYWORDS = (
    "floor",
    "ground",
    "road",
    "sidewalk",
    "pavement",
    "path",
    "earth",
    "grass",
    "field",
    "terrain",
)


def _load_array(path: Path):
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else list(data.keys())[0]
        return data[key]
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        flag = cv2.IMREAD_UNCHANGED
        arr = cv2.imread(str(path), flag)
        if arr is None:
            raise FileNotFoundError(path)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    raise ValueError(f"Unsupported array path: {path}")


class _FrameArraySource:
    def __init__(self, path: Optional[str | Path]):
        self.path = Path(path) if path else None
        self.stack = None
        if self.path is not None and self.path.is_file():
            self.stack = _load_array(self.path)
            if isinstance(self.stack, torch.Tensor):
                self.stack = self.stack.detach().cpu().numpy()

    def get(self, index: int):
        if self.path is None:
            return None
        if self.stack is not None:
            return np.asarray(self.stack[index])
        candidates = [
            self.path / f"{index:06d}.pt",
            self.path / f"{index:06d}.npy",
            self.path / f"{index:06d}.npz",
            self.path / f"{index:06d}.png",
            self.path / f"{index}.pt",
            self.path / f"{index}.npy",
            self.path / f"{index}.npz",
            self.path / f"{index}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                arr = _load_array(candidate)
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                return np.asarray(arr)
        return None


def _read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    ok, frame_bgr = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ok, frame_bgr = cap.read()
    cap.release()
    return frames


def _device_index(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return -1


def _build_pipeline(task, model, device):
    if not model:
        return None
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency path
        Log.warning(f"[Ground] transformers pipeline unavailable: {exc}")
        return None
    Log.info(f"[Ground] Loading {task} model: {model}")
    return pipeline(task, model=model, device=_device_index(device))


def _resize_to(arr, width, height, interpolation):
    arr = np.asarray(arr)
    if arr.shape[:2] == (height, width):
        return arr
    return cv2.resize(arr, (width, height), interpolation=interpolation)


def _depth_from_pipeline(depth_pipe, frame_rgb, width, height):
    if depth_pipe is None:
        return None
    from PIL import Image

    pred = depth_pipe(Image.fromarray(frame_rgb))
    depth = pred.get("predicted_depth", None) if isinstance(pred, dict) else None
    if depth is not None:
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().float().cpu().numpy()
        depth = np.asarray(depth).squeeze()
    elif isinstance(pred, dict) and "depth" in pred:
        depth = np.asarray(pred["depth"]).astype(np.float32)
    else:
        return None
    depth = _resize_to(depth.astype(np.float32), width, height, cv2.INTER_LINEAR)
    return depth


def _mask_from_pipeline(seg_pipe, frame_rgb, width, height):
    if seg_pipe is None:
        return None
    from PIL import Image

    preds = seg_pipe(Image.fromarray(frame_rgb))
    if isinstance(preds, dict):
        preds = [preds]
    mask = np.zeros((height, width), dtype=bool)
    for pred in preds or []:
        label = str(pred.get("label", "")).lower()
        if not any(key in label for key in GROUND_LABEL_KEYWORDS):
            continue
        pred_mask = pred.get("mask", None)
        if pred_mask is None:
            continue
        pred_mask = np.asarray(pred_mask)
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[..., 0]
        pred_mask = _resize_to(pred_mask, width, height, cv2.INTER_NEAREST)
        mask |= pred_mask > 0
    return mask if mask.any() else None


def _bottom_ground_prior(width, height, bbx_xyxy=None):
    mask = np.zeros((height, width), dtype=bool)
    mask[int(height * 0.55) :, :] = True
    if bbx_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in bbx_xyxy]
        pad_x = int(max(8, 0.08 * (x2 - x1 + 1)))
        pad_y = int(max(8, 0.08 * (y2 - y1 + 1)))
        x1 = max(0, x1 - pad_x)
        x2 = min(width - 1, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(height - 1, y2 + pad_y)
        mask[y1 : y2 + 1, x1 : x2 + 1] = False
    return mask


def _to_bool_mask(mask, width, height):
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = _resize_to(mask, width, height, cv2.INTER_NEAREST)
    return mask > 0


def _to_depth(depth, width, height):
    if depth is None:
        return None
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = _resize_to(depth.astype(np.float32), width, height, cv2.INTER_LINEAR)
    return depth


def _points_from_depth_mask(depth, mask, K, T_w2c, max_points, rng):
    height, width = depth.shape[:2]
    valid = mask & np.isfinite(depth) & (depth > 1e-6)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return None
    if len(xs) > max_points:
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[keep]
        ys = ys[keep]

    z = depth[ys, xs].astype(np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (xs.astype(np.float64) - cx) / max(fx, 1e-8) * z
    y = (ys.astype(np.float64) - cy) / max(fy, 1e-8) * z
    points_c = np.stack([x, y, z], axis=-1)

    R = np.asarray(T_w2c[:3, :3], dtype=np.float64)
    t = np.asarray(T_w2c[:3, 3], dtype=np.float64)
    points_w = (R.T @ (points_c - t).T).T
    return points_w.astype(np.float32)


def _fit_plane_svd(points, up_hint):
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    if np.dot(normal, up_hint) < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), np.float32(offset)


def _fit_plane_ransac(points, up_hint=None, num_iterations=128, threshold=None, min_inlier_ratio=0.25, rng=None):
    if points is None or len(points) < 32:
        return None
    if rng is None:
        rng = np.random.default_rng(0)
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    up_hint = up_hint / max(np.linalg.norm(up_hint), 1e-8)

    points64 = np.asarray(points, dtype=np.float64)
    if threshold is None:
        depth_scale = np.nanmedian(np.linalg.norm(points64, axis=-1))
        threshold = max(0.02, 0.01 * float(depth_scale))

    best_mask = None
    best_count = 0
    n = len(points64)
    for _ in range(num_iterations):
        ids = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points64[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        if np.dot(normal, up_hint) < 0:
            normal = -normal
        offset = -np.dot(normal, p0)
        dist = np.abs(points64 @ normal + offset)
        mask = dist < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < max(16, int(min_inlier_ratio * n)):
        return None
    return _fit_plane_svd(points64[best_mask], up_hint)


def _fill_missing_planes(normals, offsets, valid):
    length = len(valid)
    if not valid.any():
        return normals, offsets, valid
    valid_ids = np.flatnonzero(valid)
    for i in range(length):
        if valid[i]:
            continue
        nearest = valid_ids[np.argmin(np.abs(valid_ids - i))]
        normals[i] = normals[nearest]
        offsets[i] = offsets[nearest]
    return normals, offsets, valid


def _smooth_planes(normals, offsets, window=5):
    if len(normals) < 3 or window <= 1:
        return normals, offsets
    window = min(window, len(normals))
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)

    padded_normals = np.pad(normals, ((pad, pad), (0, 0)), mode="edge")
    smoothed_normals = np.stack(
        [np.convolve(padded_normals[:, c], kernel, mode="valid") for c in range(3)], axis=-1
    )
    smoothed_normals /= np.maximum(np.linalg.norm(smoothed_normals, axis=-1, keepdims=True), 1e-8)
    flip = smoothed_normals[:, 1] < 0
    smoothed_normals[flip] *= -1
    offsets = np.convolve(np.pad(offsets, (pad, pad), mode="edge"), kernel, mode="valid")
    return smoothed_normals.astype(np.float32), offsets.astype(np.float32)


@torch.no_grad()
def estimate_ground_plane_from_video(
    video_path,
    K_fullimg,
    T_w2c,
    bbx_xyxy=None,
    static_camera=False,
    depth_model=None,
    seg_model=None,
    depth_path=None,
    mask_path=None,
    output_path=None,
    device="cuda:0",
    stride=1,
    max_points_per_frame=5000,
    min_points=128,
    use_depth_offset=True,
):
    """Estimate ground plane normals/offsets from depth and visible-ground masks.

    Args:
        video_path: Source video.
        K_fullimg: Tensor/array shaped (L, 3, 3).
        T_w2c: Tensor/array shaped (L, 4, 4).
        bbx_xyxy: Optional human boxes shaped (L, 4), used to remove the person
            from the fallback ground mask.
        depth_model/seg_model: Optional transformers model names or local paths.
        depth_path/mask_path: Optional precomputed stack file or frame directory.
        use_depth_offset: If False, save only normals. Use False for relative-depth
            models when their metric scale is not trustworthy.
    """
    frames = _read_video_frames(video_path)
    if len(frames) == 0:
        Log.warning("[Ground] Empty video; skipping ground-plane estimation.")
        return None

    K_fullimg = torch.as_tensor(K_fullimg).detach().cpu().float().numpy()
    T_w2c = torch.as_tensor(T_w2c).detach().cpu().float().numpy()
    if bbx_xyxy is not None:
        bbx_xyxy = torch.as_tensor(bbx_xyxy).detach().cpu().float().numpy()

    depth_source = _FrameArraySource(depth_path)
    mask_source = _FrameArraySource(mask_path)
    depth_pipe = _build_pipeline("depth-estimation", depth_model, device) if depth_path is None else None
    seg_pipe = _build_pipeline("image-segmentation", seg_model, device) if mask_path is None else None

    if depth_source.path is None and depth_pipe is None:
        Log.warning("[Ground] No depth source configured; skipping ground-plane estimation.")
        return None
    if mask_source.path is None and seg_pipe is None:
        Log.warning("[Ground] No segmentation source configured; using bottom-image ground prior.")

    length = min(len(frames), len(K_fullimg), len(T_w2c))
    stride = max(1, int(stride))
    rng = np.random.default_rng(7)
    normals = np.zeros((length, 3), dtype=np.float32)
    offsets = np.zeros((length,), dtype=np.float32)
    valid = np.zeros((length,), dtype=bool)

    frame_points = {}
    iterator = range(0, length, stride)
    for i in tqdm(iterator, desc="Ground plane", leave=False):
        frame = frames[i]
        height, width = frame.shape[:2]
        depth = _to_depth(depth_source.get(i), width, height)
        if depth is None:
            depth = _depth_from_pipeline(depth_pipe, frame, width, height)
        if depth is None:
            continue

        mask = _to_bool_mask(mask_source.get(i), width, height)
        if mask is None:
            mask = _mask_from_pipeline(seg_pipe, frame, width, height)
        if mask is None:
            box = bbx_xyxy[i] if bbx_xyxy is not None and i < len(bbx_xyxy) else None
            mask = _bottom_ground_prior(width, height, box)

        points = _points_from_depth_mask(depth, mask, K_fullimg[i], T_w2c[i], max_points_per_frame, rng)
        if points is None or len(points) < min_points:
            continue
        frame_points[i] = points

    if static_camera:
        if not frame_points:
            Log.warning("[Ground] Could not collect enough ground/depth points.")
            return None
        all_points = np.concatenate(list(frame_points.values()), axis=0)
        if len(all_points) > max_points_per_frame * 4:
            keep = rng.choice(len(all_points), size=max_points_per_frame * 4, replace=False)
            all_points = all_points[keep]
        fit = _fit_plane_ransac(all_points, rng=rng)
        if fit is None:
            Log.warning("[Ground] Static ground plane RANSAC failed.")
            return None
        normal, offset = fit
        normals[:] = normal
        offsets[:] = offset
        valid[:] = True
    else:
        for i, points in frame_points.items():
            fit = _fit_plane_ransac(points, rng=rng)
            if fit is None:
                continue
            normals[i], offsets[i] = fit
            valid[i] = True
        if not valid.any():
            Log.warning("[Ground] Per-frame ground plane RANSAC failed.")
            return None
        normals, offsets, valid = _fill_missing_planes(normals, offsets, valid)
        normals, offsets = _smooth_planes(normals, offsets)

    result = {
        "ground_normal_world": torch.from_numpy(normals).float(),
        "ground_valid": torch.from_numpy(valid),
        "meta": {
            "source": "segmentation_depth",
            "depth_model": str(depth_model) if depth_model else None,
            "seg_model": str(seg_model) if seg_model else None,
            "depth_path": str(depth_path) if depth_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "use_depth_offset": bool(use_depth_offset),
            "static_camera": bool(static_camera),
        },
    }
    if use_depth_offset:
        result["ground_plane_offset"] = torch.from_numpy(offsets).float()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        Log.info(f"[Ground] Saved ground plane to {output_path}")
    return result
