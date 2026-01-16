from typing import Optional, Dict, Tuple

import torch
import numpy as np
import einops

from robosuite.utils import camera_utils as CU
from lerobot.common.datasets.camera_utils import (
    PluckerEmbedder,
    remove_extrinsic_camera_axis_correction
)
from lerobot.common.datasets.viz_utils import (
    _get_motion_dynamics_basis,
    _make_motion_basis_axis_rgb_tensor_cam_to_world,
    save_rgb_image,
    _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world,
)

def _get_intrinsic_and_extrinsic(
    env, camera_name, height, width, batchwise: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    intrinsic_matrix = torch.from_numpy(CU.get_camera_intrinsic_matrix(env.sim, camera_name, height, width)).float()

    extrinsic_matrix = torch.from_numpy(CU.get_camera_extrinsic_matrix(env.sim, camera_name)).float()
    extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)

    if batchwise:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0)
        extrinsic_matrix = extrinsic_matrix.unsqueeze(0)

    return intrinsic_matrix, extrinsic_matrix

def _calculate_plucker_tensor(intrinsic_matrix: torch.Tensor, extrinsic_matrix: torch.Tensor, height: int):
    plucker_embedder = PluckerEmbedder(img_size=height, device='cpu')

    with torch.no_grad():
        plucker_data = plucker_embedder(intrinsic_matrix, extrinsic_matrix)
        plucker_tensor = einops.rearrange(plucker_data['plucker'], 's h w c -> s c h w').to('cuda', non_blocking=True)

    return plucker_tensor

def _calculate_axis_tensor(state, intrinsic_matrix, extrinsic_matrix, img):
    with torch.no_grad():
        rgb_tensor = einops.rearrange(torch.Tensor(img), 'h w c -> c h w')
        rgb_tensor = rgb_tensor.unsqueeze(0)
        rgb_tensor /= 255

        motion_dynamics_basis = _get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=extrinsic_matrix).reshape(-1)

        axis_tensor, origin_xy = _make_motion_basis_axis_rgb_tensor_cam_to_world(
            rgb_tensor=rgb_tensor.to('cpu'),                # (1, 3, 224, 224)
            motion_dynamics_basis=motion_dynamics_basis,
            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
            intrinsic_matrix=intrinsic_matrix,
            robot_eef_abs_poses=state[:7],  # eef pose (7,)
            origin_robot=True,
            origin_fallback="pp",
            arrow_len=60,
            return_overlay=False,
        )
    return axis_tensor

def _calculate_rescaled_axis_tensor(state, intrinsic_matrix, extrinsic_matrix, img):
    with torch.no_grad():
        rgb_tensor = einops.rearrange(torch.Tensor(img), 'h w c -> c h w')
        rgb_tensor = rgb_tensor.unsqueeze(0)
        rgb_tensor /= 255.0

        axis_tensor, origin_xy = _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world(
            rgb_tensor=rgb_tensor.to('cpu'),                  # (1, 3, H, W)
            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
            intrinsic_matrix=intrinsic_matrix,
            robot_eef_abs_poses=state[:7],  # eef pose (7)
            origin_robot=True,
            origin_fallback="pp",
            arrow_len=60,
            return_overlay=False,
        )
        # save_rgb_image(axis_tensor, "basis.png")
    return axis_tensor

def _calculate_vla_additional_inputs(
    vla_mode: str,
    intrinsic_matrix: torch.Tensor,
    extrinsic_matrix: torch.Tensor,
    img: Optional[np.ndarray],
    state: Optional[np.ndarray]
) -> Dict[str, torch.Tensor] :
    if vla_mode == "plucker":
        return {"plucker": _calculate_plucker_tensor(intrinsic_matrix, extrinsic_matrix, img.shape[0])}

    if vla_mode == "plucker_concat":
        return {"concat": _calculate_plucker_tensor(intrinsic_matrix, extrinsic_matrix, img.shape[0])}

    if vla_mode == "basis":
        return {"basis": _calculate_axis_tensor(state, intrinsic_matrix, extrinsic_matrix, img)}

    if vla_mode == "basis_rescale":
        return {"basis": _calculate_rescaled_axis_tensor(state, intrinsic_matrix, extrinsic_matrix, img)}

    if vla_mode == "basis_rescale_concat":
        return {"concat": _calculate_rescaled_axis_tensor(state, intrinsic_matrix, extrinsic_matrix, img)}

    return {}
