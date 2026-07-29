# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact-based postprocessing for global body parameters.

Refines predicted global translation by penalizing motion at joints predicted
to be in contact, and optionally adjusts body pose via CCD inverse kinematics
to enforce contact constraints.
"""

import torch
from torch.cuda.amp import autocast

import gem.utils.matrix as matrix
from gem.network.endecoder import EnDecoder
from gem.utils.ccd_ik import CCD_IK
from gem.utils.net_utils import gaussian_smooth
from gem.utils.rotation_conversions import matrix_to_axis_angle

# SOMA77 contact joint indices: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
SOMA77_CONTACT_JOINT_IDS = [69, 70, 74, 75, 14, 42]
SOMA77_FOOT_CONTACT_JOINT_IDS = [69, 70, 74, 75]

# SOMA77 kinematic chain definitions for CCD IK
SOMA77_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
]
SOMA77_LEFT_LEG_CHAIN = [0, 67, 68, 69, 70]
SOMA77_RIGHT_LEG_CHAIN = [0, 72, 73, 74, 75]
SOMA77_LEFT_HAND_CHAIN = [3, 11, 12, 13, 14]
SOMA77_RIGHT_HAND_CHAIN = [3, 39, 40, 41, 42]
SOMA77_LEFT_FOOT_IDS = [69, 70]
SOMA77_RIGHT_FOOT_IDS = [74, 75]
SOMA77_LEFT_WRIST_IDS = [14]
SOMA77_RIGHT_WRIST_IDS = [42]


def _as_batched_ground_normal(ground_normal_world, reference_points):
    """Return normalized ground normal(s) shaped (B, 3) or (B, L, 3)."""
    batch_size = reference_points.shape[0]
    seq_len = reference_points.shape[1] if reference_points.ndim >= 4 else None
    if ground_normal_world is None:
        normal = reference_points.new_tensor([0.0, 1.0, 0.0]).expand(batch_size, -1)
    else:
        normal = torch.as_tensor(
            ground_normal_world,
            dtype=reference_points.dtype,
            device=reference_points.device,
        )
        if normal.ndim == 1:
            normal = normal.unsqueeze(0).expand(batch_size, -1)
        elif normal.ndim == 2 and normal.shape[0] == seq_len and batch_size == 1:
            normal = normal.unsqueeze(0)
        elif normal.ndim in (2, 3) and normal.shape[0] != batch_size:
            raise ValueError(
                f"ground_normal_world batch mismatch: expected {batch_size}, got {normal.shape[0]}"
            )
        elif normal.ndim not in (2, 3):
            raise ValueError(
                "ground_normal_world must be shaped (3,), (B, 3), (L, 3), or (B, L, 3)"
            )
        if normal.ndim == 3 and seq_len is not None and normal.shape[1] != seq_len:
            raise ValueError(
                f"ground_normal_world length mismatch: expected {seq_len}, got {normal.shape[1]}"
            )

    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(normal[..., 1:2] < 0, -normal, normal)


def _as_batched_plane_offset(ground_plane_offset, reference_points):
    """Return plane offset d for n dot p + d = 0, shaped (B,) or (B, L)."""
    batch_size = reference_points.shape[0]
    seq_len = reference_points.shape[1] if reference_points.ndim >= 4 else None
    if ground_plane_offset is None:
        return reference_points.new_zeros(batch_size)

    offset = torch.as_tensor(
        ground_plane_offset,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    if offset.ndim == 0:
        return offset.expand(batch_size)
    if offset.ndim == 1 and offset.shape[0] == batch_size:
        return offset
    if offset.ndim == 1 and offset.shape[0] == seq_len and batch_size == 1:
        return offset.unsqueeze(0)
    if offset.ndim == 2 and offset.shape[:2] == (batch_size, seq_len):
        return offset
    raise ValueError(
        f"ground_plane_offset mismatch: expected scalar, ({batch_size},), or ({batch_size}, {seq_len}), got {tuple(offset.shape)}"
    )


def signed_distance_to_ground_plane(
    points,
    ground_normal_world=None,
    ground_plane_offset=None,
):
    """Signed distance from points to the plane n dot p + d = 0.

    Args:
        points: Tensor shaped (B, ..., 3), usually (B, L, J, 3).
        ground_normal_world: Optional normal shaped (3,), (B, 3), or (B, L, 3).
        ground_plane_offset: Optional plane offset d shaped scalar, (B,), or (B, L).

    Returns:
        Signed distances shaped (B, ...), plus normalized normals.
    """
    normal = _as_batched_ground_normal(ground_normal_world, points)
    offset = _as_batched_plane_offset(ground_plane_offset, points)
    if normal.ndim == 2:
        distances = torch.einsum("b...c,bc->b...", points, normal)
        if offset.ndim == 1:
            offset = offset.view([points.shape[0]] + [1] * (points.ndim - 2))
        else:
            offset = offset.view([points.shape[0], points.shape[1]] + [1] * (points.ndim - 3))
    else:
        distances = torch.einsum("bl...c,blc->bl...", points, normal)
        if offset.ndim == 1:
            offset = offset[:, None].expand(-1, points.shape[1])
        offset = offset.view([points.shape[0], points.shape[1]] + [1] * (points.ndim - 3))
    return distances + offset, normal


def shift_translation_to_ground_plane(
    transl,
    joints,
    ground_normal_world=None,
    ground_plane_offset=None,
    contact_logits=None,
):
    """Shift translation so the sequence touches, but airborne frames stay airborne.

    The grounding distance is one scalar per sequence, estimated from contact
    joints when contact logits are provided. A frame-level normal (B, L, 3) uses
    a framewise correction direction with that same sequence-level distance.
    """
    distances, normal = signed_distance_to_ground_plane(
        joints,
        ground_normal_world=ground_normal_world,
        ground_plane_offset=ground_plane_offset,
    )
    all_min_distance = distances.flatten(1).min(dim=-1)[0]
    min_distance = all_min_distance

    if contact_logits is not None and contact_logits.shape == distances.shape:
        contact_mask = contact_logits > 0
        has_contact = contact_mask.flatten(1).any(dim=-1)
        masked_distances = distances.masked_fill(~contact_mask, float("inf"))
        contact_min_distance = masked_distances.flatten(1).min(dim=-1)[0]
        min_distance = torch.where(has_contact, contact_min_distance, all_min_distance)

    if normal.ndim == 2:
        correction = -min_distance[:, None] * normal
        return transl + correction[:, None, :]

    correction = -min_distance[:, None, None] * normal
    return transl + correction


def _estimate_plane_offset_from_contacts(points, normal, contact_logits=None):
    """Estimate d in n dot p + d = 0 from current contact-foot positions."""
    if normal.ndim == 2:
        distances = torch.einsum("bljc,bc->blj", points, normal)
    else:
        distances = torch.einsum("bljc,blc->blj", points, normal)

    all_min = distances.flatten(1).min(dim=-1)[0]
    if contact_logits is None or contact_logits.shape != distances.shape:
        if normal.ndim == 2:
            return -all_min
        return -all_min[:, None].expand(-1, points.shape[1])

    contact_mask = contact_logits > 0
    masked = distances.masked_fill(~contact_mask, float("inf"))
    has_any_contact = contact_mask.flatten(1).any(dim=-1)

    if normal.ndim == 2:
        contact_min = masked.flatten(1).min(dim=-1)[0]
        plane_level = torch.where(has_any_contact, contact_min, all_min)
        return -plane_level

    has_frame_contact = contact_mask.any(dim=-1)
    frame_min = masked.min(dim=-1)[0]
    fallback = all_min[:, None].expand(-1, points.shape[1])
    plane_level = torch.where(has_frame_contact, frame_min, fallback)
    return -plane_level


def _ray_plane_intersection_world(kp2d, K_fullimg, T_w2c, normal, offset, eps=1e-5):
    """Back-project 2D pixels to 3D by intersecting camera rays with a ground plane."""
    B, L, J = kp2d.shape[:3]
    if T_w2c is None:
        T_w2c = torch.eye(4, dtype=kp2d.dtype, device=kp2d.device).reshape(1, 1, 4, 4)
        T_w2c = T_w2c.expand(B, L, -1, -1)

    uv1 = torch.cat([kp2d[..., :2], torch.ones_like(kp2d[..., :1])], dim=-1).float()
    K_inv = torch.linalg.inv(K_fullimg.float())
    ray_c = torch.einsum("blij,blnj->blni", K_inv, uv1)
    ray_c = ray_c / ray_c.norm(dim=-1, keepdim=True).clamp_min(eps)

    R_w2c = T_w2c[..., :3, :3].float()
    t_w2c = T_w2c[..., :3, 3].float()
    R_c2w = R_w2c.transpose(-1, -2)
    origin_w = -torch.einsum("blij,blj->bli", R_c2w, t_w2c)
    ray_w = torch.einsum("blij,blnj->blni", R_c2w, ray_c)
    ray_w = ray_w / ray_w.norm(dim=-1, keepdim=True).clamp_min(eps)

    if normal.ndim == 2:
        n = normal[:, None, None, :].float()
        d = offset[:, None, None].float()
    else:
        n = normal[:, :, None, :].float()
        if offset.ndim == 1:
            offset = offset[:, None].expand(-1, L)
        d = offset[:, :, None].float()

    plane_at_origin = (origin_w[:, :, None, :] * n).sum(dim=-1) + d
    denom = (ray_w * n).sum(dim=-1)
    safe_denom = torch.where(denom >= 0, denom.clamp_min(eps), denom.clamp_max(-eps))
    ray_depth = -plane_at_origin / safe_denom
    valid = denom.abs() > eps
    valid = valid & torch.isfinite(ray_depth) & (ray_depth > 0)
    target_w = origin_w[:, :, None, :] + ray_depth[..., None] * ray_w
    return target_w.to(kp2d.dtype), valid, ray_depth.to(kp2d.dtype)


def refine_translation_with_ground_ray_contacts(
    transl,
    joints,
    kp2d=None,
    K_fullimg=None,
    T_w2c=None,
    ground_normal_world=None,
    ground_plane_offset=None,
    contact_logits=None,
    kp_conf_thr=0.4,
    blend=0.5,
    max_correction=0.75,
):
    """Refine root translation from 2D foot rays intersected with the ground plane."""
    if kp2d is None or K_fullimg is None:
        return transl
    if kp2d.ndim != 4 or kp2d.shape[-1] < 2:
        return transl
    if K_fullimg.ndim != 4 or K_fullimg.shape[-2:] != (3, 3):
        return transl
    if kp2d.shape[-2] <= max(SOMA77_FOOT_CONTACT_JOINT_IDS):
        return transl

    foot_ids = SOMA77_FOOT_CONTACT_JOINT_IDS
    foot_joints = joints[:, :, foot_ids]
    foot_kp2d = kp2d[:, :, foot_ids].to(transl)
    normal = _as_batched_ground_normal(ground_normal_world, foot_joints)

    foot_logits = None
    if contact_logits is not None and contact_logits.shape[-1] >= len(foot_ids):
        foot_logits = contact_logits[:, :, : len(foot_ids)].to(transl)

    if ground_plane_offset is None:
        offset = _estimate_plane_offset_from_contacts(foot_joints, normal, foot_logits)
    else:
        offset = _as_batched_plane_offset(ground_plane_offset, foot_joints)

    target_w, ray_valid, _ = _ray_plane_intersection_world(
        foot_kp2d,
        K_fullimg.to(transl),
        T_w2c.to(transl) if T_w2c is not None else None,
        normal,
        offset,
    )

    conf = foot_kp2d[..., 2] if foot_kp2d.shape[-1] > 2 else torch.ones_like(ray_valid, dtype=transl.dtype)
    weights = (conf > kp_conf_thr).to(transl) * conf.clamp(min=0.0, max=1.0)
    if foot_logits is not None:
        contact_gate = (foot_logits > 0).to(transl)
        weights = weights * contact_gate * foot_logits.sigmoid()
    weights = weights * ray_valid.to(transl)

    weight_sum = weights.sum(dim=-1, keepdim=True)
    has_target = weight_sum.squeeze(-1) > 1e-6
    if not has_target.any():
        return transl

    foot_offsets = foot_joints - transl.unsqueeze(-2)
    target_transl_per_joint = target_w - foot_offsets
    target_transl = (target_transl_per_joint * weights[..., None]).sum(dim=-2)
    target_transl = target_transl / weight_sum.clamp_min(1e-6)

    correction = target_transl - transl
    correction = torch.where(has_target[..., None], correction, torch.zeros_like(correction))
    corr_norm = correction.norm(dim=-1, keepdim=True)
    correction = correction * (max_correction / corr_norm.clamp_min(max_correction)).clamp(max=1.0)
    correction = correction * blend

    # Spread depth/lateral correction softly to neighboring frames, but keep jump height local.
    correction[..., 0] = gaussian_smooth(correction[..., 0], dim=-1)
    correction[..., 2] = gaussian_smooth(correction[..., 2], dim=-1)
    return transl + correction


@autocast(enabled=False)
def refine_translation_with_contacts(
    outputs, endecoder: EnDecoder, smpl_key="pred_body_params_global"
):
    """Refine global translation using predicted contact labels.

    For joints predicted to be in contact (static), their inter-frame displacement
    is used to correct the global translation via a softmax-weighted scheme.
    The result is smoothed (x, z) and grounded against n dot p + d = 0.

    Args:
        outputs: dict with 'static_conf_logits' and smpl_key body params.
        endecoder: EnDecoder with fk_v2() support.
        smpl_key: key into outputs for the body params dict.

    Returns:
        Refined translation tensor (B, L, 3).
    """
    joint_ids = SOMA77_CONTACT_JOINT_IDS

    # Global FK to get joint positions
    pred_w_j3d = endecoder.fk_v2(**outputs[smpl_key])
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J_contact, 3)

    # Compute per-frame displacement of contact joints
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J_contact, 3)

    # Process contact logits: softmax-weighted displacement of static joints
    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label = static_conf_logits > 0  # (B, L-1, J_contact)
    # Mask out non-contact logits before softmax (fp16-safe)
    static_conf_logits = static_conf_logits.float() - (~static_label * 1e6)
    is_static = static_label.sum(dim=-1) > 0  # (B, L-1)

    # Weighted displacement: softmax across joints, zero if no joints static
    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(
        dim=-2
    )  # (B, L-1, J_contact, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J_contact, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)

    # Correct translation via vectorized cumsum
    pred_w_transl = outputs[smpl_key]["transl"].clone()  # (B, L, 3)
    pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
    corrected_disp = pred_w_disp - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], corrected_disp], dim=1), dim=1)
    # Smooth x and z components
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # First use 2D foot rays + ground plane to correct full 3D translation/depth.
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    post_w_transl = refine_translation_with_ground_ray_contacts(
        post_w_transl,
        post_w_j3d,
        kp2d=outputs.get("kp2d", None),
        K_fullimg=outputs.get("K_fullimg", None),
        T_w2c=outputs.get("T_w2c", None),
        ground_normal_world=outputs.get("ground_normal_world", None),
        ground_plane_offset=outputs.get("ground_plane_offset", None),
        contact_logits=outputs.get("static_conf_logits", None),
    )

    # Ground the sequence using signed distance to n dot p + d = 0.
    # Defaults to the old horizontal plane y = 0 when no plane is provided.
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    post_w_transl = shift_translation_to_ground_plane(
        post_w_transl,
        post_w_j3d[:, :, joint_ids],
        ground_normal_world=outputs.get("ground_normal_world", None),
        ground_plane_offset=outputs.get("ground_plane_offset", None),
        contact_logits=outputs.get("static_conf_logits", None),
    )

    return post_w_transl


@autocast(enabled=False)
def refine_pose_with_contact_ik(
    outputs, endecoder: EnDecoder, static_conf=None, smpl_key="pred_body_params_global"
):
    """Refine body pose via CCD inverse kinematics to enforce contact constraints.

    Interpolates contact joint target positions based on predicted confidence,
    then solves IK on 4 kinematic chains (2 legs, 2 arms) to reach those targets.

    Args:
        outputs: dict with body params and static_conf_logits.
        endecoder: EnDecoder with fk_v2() support.
        static_conf: optional pre-computed contact confidence (B, L, J). If None,
            sigmoid of static_conf_logits is used.
        smpl_key: key into outputs for the body params dict.

    Returns:
        Refined body_pose tensor (B, L, (J-1)*3) in axis-angle.
    """
    if static_conf is None:
        static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)

    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs[smpl_key], get_intermediate=True)

    joint_ids = SOMA77_CONTACT_JOINT_IDS
    parents = SOMA77_PARENTS

    # Interpolate contact joint targets: blend previous-frame position with current prediction
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    global_rot = matrix.get_rotation(post_w_mat)

    def _solve_chain_ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )
        chain_local_mat = solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    # Solve IK for all 4 chains
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_LEFT_FOOT_IDS],
        global_rot[:, :, SOMA77_LEFT_FOOT_IDS],
        [3],
        SOMA77_LEFT_LEG_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_RIGHT_FOOT_IDS],
        global_rot[:, :, SOMA77_RIGHT_FOOT_IDS],
        [3],
        SOMA77_RIGHT_LEG_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_LEFT_WRIST_IDS],
        global_rot[:, :, SOMA77_LEFT_WRIST_IDS],
        [4],
        SOMA77_LEFT_HAND_CHAIN,
    )
    local_mat = _solve_chain_ik(
        local_mat,
        post_target_j3d[:, :, SOMA77_RIGHT_WRIST_IDS],
        global_rot[:, :, SOMA77_RIGHT_WRIST_IDS],
        [4],
        SOMA77_RIGHT_HAND_CHAIN,
    )

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose
