from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from ext_roban.tasks.manager_based.amp.envs import ManagerBasedRLAMPEnv


def tangent_and_normal(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = math_utils.quat_rotate(q, ref_tangent)
    normal = math_utils.quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


def rel_key_body_pos(env: ManagerBasedRLAMPEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    key_body_positions = asset.data.body_pos_w[:, env.key_body_indexes]
    root_positions = asset.data.body_pos_w[:, env.ref_body_index]
    return (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1)
