from __future__ import annotations

import math
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from ext_roban.tasks.manager_based.amp.envs import ManagerBasedRLAMPEnv



def reset_scene_strategy_random(env: ManagerBasedRLAMPEnv,
                                env_ids: torch.Tensor,
                                start: bool = False,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    # sample random motion times (or zeros if start is True)
    num_samples = env_ids.shape[0]
    times = np.zeros(num_samples) if start else env._motion_loader.sample_times(num_samples)
    # sample random motions
    (
        dof_positions,
        dof_velocities,
        body_positions,
        body_rotations,
        body_linear_velocities,
        body_angular_velocities,
    ) = env._motion_loader.sample(num_samples=num_samples, times=times)

    # get root transforms (the humanoid torso)
    motion_torso_index = env._motion_loader.get_body_index(["base_link"])[0]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = body_positions[:, motion_torso_index] + env.scene.env_origins[env_ids]
    root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
    root_state[:, 3:7] = body_rotations[:, motion_torso_index]
    root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
    root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
    # get DOFs state
    dof_pos = dof_positions[:, env.motion_dof_indexes]
    dof_vel = dof_velocities[:, env.motion_dof_indexes]

    # update AMP observation
    amp_observations = env.collect_reference_motions(num_samples, times)
    env.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, env.cfg.num_amp_observations, -1)

    if torch.any(torch.isnan(root_state)):
        print(f"reset random root_state have nan: {root_state}")
    
    asset.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)








