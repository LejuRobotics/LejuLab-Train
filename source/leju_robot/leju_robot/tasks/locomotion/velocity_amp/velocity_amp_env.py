# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np
import torch
import random
import math
# from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse
from isaaclab.assets.articulation import Articulation

from leju_robot.tasks.amp.envs import ManagerBasedRLAMPEnv


class LocomotionVelocityAMPEnv(ManagerBasedRLAMPEnv):

    def load_managers(self):
        """Load managers and initialize related variables."""
        # Call parent load_managers first so all managers are loaded.
        super().load_managers()

        # Get robot joint count.
        robot: Articulation = self.scene["robot"]
        num_joints = robot.num_joints

        self.ref_stand_pos = torch.zeros((self.num_envs, num_joints), device=self.device)

        # Initialize reference joint positions (for error computation).
        self.ref_dof_pos = robot.data.default_joint_pos.clone()

        # Initialize reference actions.
        self.ref_action = torch.zeros((self.num_envs, num_joints), device=self.device)

        self.isPlay = False

        # Action term scale and offset (for inverse mapping from commanded positions).
        action_term = self.action_manager._terms["joint_pos"]
        self.action_scale = action_term._scale
        self.action_offset = action_term._offset

    def compute_ref_state(self):
        # Get robot default joint positions (same shape as joint vector).
        robot: Articulation = self.scene["robot"]

        # Zero ref_stand_pos, then fill standing pose targets per joint index.
        self.ref_stand_pos = robot.data.default_joint_pos.clone() * 0
        self.ref_stand_pos[:, 1] = -0.05033
        self.ref_stand_pos[:, 2] = -0.0164
        self.ref_stand_pos[:, 3] = -0.0233
        self.ref_stand_pos[:, 4] = 0.155
        self.ref_stand_pos[:, 5] = -0.1

        self.ref_stand_pos[:, 7] = 0.05033
        self.ref_stand_pos[:, 8] = 0.0164
        self.ref_stand_pos[:, 9] = -0.0233
        self.ref_stand_pos[:, 10] = 0.155
        self.ref_stand_pos[:, 11] = -0.1


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        return super().step(action)


    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples,num_observation=self.cfg.num_amp_observations)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        return amp_observation.view(-1, self.amp_observation_size)

    def collect_reference_motions_by_motion_idx(self, num_samples: int, current_times: np.ndarray | None = None,motion_idx = -1) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample_by_motion_idx(num_samples=num_samples, times=times,motion_idx=motion_idx)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size),body_positions, body_rotations, body_linear_velocities, body_angular_velocities,times


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    gravity_vec = torch.zeros_like(root_rotations[..., :3])
    gravity_vec[..., -1] = -1.
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            # quaternion_to_tangent_and_normal(root_rotations),
            quat_rotate_inverse(root_rotations, gravity_vec),
            quat_rotate_inverse(root_rotations, root_linear_velocities),
            quat_rotate_inverse(root_rotations, root_angular_velocities),
            quat_rotate_inverse(root_rotations.unsqueeze(-2), key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
