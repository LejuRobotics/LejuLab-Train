from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_yaw
from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_feet_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward

def motion_feet_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel

def hand_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when contact force times velocity exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    velocity_norm = torch.norm(body_vel, dim=-1)
    force_velocity_product = contact_force_norm * velocity_norm
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)
    return reward

def body_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 300.0,
) -> torch.Tensor:
    """Penalize when historical maximum contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    max_contact_force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    max_force_violation = torch.clamp(max_contact_force_norm - force_threshold, min=0.0)
    max_force_penalty = torch.sum(max_force_violation / force_threshold, dim=1)
    return max_force_penalty

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when historical maximum contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] 
    velocity_norm = torch.norm(body_vel, dim=-1) 
    force_velocity_product = contact_force_norm * velocity_norm  
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  
    reward = torch.sum(violation / force_velocity_threshold, dim=1) 
    return reward

def body_contact_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize velocity when contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    max_contact_force = torch.max(net_contact_forces[:, :, sensor_cfg.body_ids, 2], dim=1)[0]
    is_contact = max_contact_force > force_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    reward = torch.sum(torch.square(body_vel_z) * is_contact, dim=1)
    return reward

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize when contact force times velocity exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    velocity_norm = torch.norm(body_vel, dim=-1)  # (num_envs, num_feet)
    force_velocity_product = contact_force_norm * velocity_norm  # (num_envs, num_feet)
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  # (num_envs, num_feet)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)  # (num_envs,)
    return reward


def motion_default_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma = float,
    delta = float,
    start_frames = int,
    end_frames = int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""Reward for maintaining default pose at trajectory start and end using exponential kernel.

    .. math:: K(x, \sigma, \delta) = \exp\left(-\left(\frac{\max(0, \|x\| - \delta)}{\sigma}\right)^2\right)
    """
    joint_pos = env.command_manager.get_command(command_name)
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    asset: Articulation = env.scene[asset_cfg.name]
    time_steps = command.time_steps
    total_steps = command.motion.time_step_total
    in_start = time_steps < start_frames
    in_end = time_steps > (total_steps - 1 - end_frames)
    in_boundary = in_start | in_end
    default_joint_pos = asset.data.default_joint_pos_nominal
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error_vec = current_joint_pos - default_joint_pos
    error_norm = torch.linalg.vector_norm(error_vec, dim=-1)
    clipped_error = torch.clamp(error_norm - delta, min=0.0)
    scaled_error_squared = torch.square(clipped_error / sigma)
    reward = torch.exp(-scaled_error_squared)
    reward = reward * in_boundary.float()
    return reward

def com_balance_when_stand(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward COM balance based on support condition."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_com_pos_w = asset.data.body_com_pos_w
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True)
    system_com = (body_com_pos_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]

    feet_max_forces = contact_forces.norm(dim=-1).max(dim=1)[0]

    left_foot_contact = (feet_max_forces[:, 0] > total_mass.squeeze(-1) * 9.8 * 0.6)
    right_foot_contact = (feet_max_forces[:, 1] > total_mass.squeeze(-1) * 9.8 * 0.6)

    total_contact_force = feet_max_forces[:, 0] + feet_max_forces[:, 1]
    both_feet_support = (total_contact_force > total_mass.squeeze(-1) * 9.8 * 0.8)

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    waist_quat = asset.data.body_quat_w[:, 0, :]
    left_foot_pos = feet_pos[:, 0, :]
    right_foot_pos = feet_pos[:, 1, :]

    foot_offset_local = torch.tensor([0.02, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    foot_offset_world = quat_apply_yaw(waist_quat, foot_offset_local)
    left_foot_center = left_foot_pos + foot_offset_world
    right_foot_center = right_foot_pos + foot_offset_world

    only_left_contact = left_foot_contact & (~right_foot_contact) & (~both_feet_support)
    only_right_contact = (~left_foot_contact) & right_foot_contact & (~both_feet_support)
    no_contact = (~left_foot_contact) & (~right_foot_contact) & (~both_feet_support)
    has_contact = ~no_contact

    both_feet_target = (left_foot_center + right_foot_center) / 2.0

    target_com = torch.where(
        both_feet_support.unsqueeze(-1),
        both_feet_target,
        torch.where(
            only_left_contact.unsqueeze(-1),
            left_foot_center,
            torch.where(
                only_right_contact.unsqueeze(-1),
                right_foot_center,
                both_feet_target
            )
        )
    )

    com_offset = torch.norm(system_com[:, :2] - target_com[:, :2], dim=1)

    reward = torch.exp(-torch.square(1.5 * com_offset / std))

    reward = torch.where(has_contact, reward, torch.zeros_like(reward))

    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # (N, history, 2, 3)
    
    # left_foot_contact = contact_forces[0] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # right_foot_contact = contact_forces[1] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # both_feet_support = left_foot_contact & right_foot_contact
    
    # left_foot_air_time = torch.where(left_foot_contact & (~right_foot_contact), left_foot_air_time + env.step_dt, 0.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    total_contact_force = torch.sum(contact_forces, dim=1)
    body_weight = total_mass * 9.8
    stand_mask = (total_contact_force > body_weight * 0.8) & (total_contact_force < body_weight * 1.2)
    # left_foot_contact = contact_forces[0] > contact_forces[1] * 2.0
    # right_foot_contact = contact_forces[1] > contact_forces[0] * 2.0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command: MotionCommand = env.command_manager.get_term(command_name)
    # mask = torch.where(command.start_time < 50, 0, 1)
    # mask = torch.where(command.out_time > 0, 1, mask)
    reward *= stand_mask # * mask
    return reward

def body_slide_vel(
        env, sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_threshold: float = 110.0
    ) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > contact_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel

def penalize_feet_dragging(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    drag_threshold: float = 5.0,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize dragging of swing foot during single foot support."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]

    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    body_weight = total_mass * 9.8

    foot_in_contact = contact_forces_z > contact_threshold

    num_feet_in_contact = torch.sum(foot_in_contact.int(), dim=1)
    single_foot_support = num_feet_in_contact == 1

    left_foot_force = contact_forces_z[:, 0]
    right_foot_force = contact_forces_z[:, 1]

    left_is_support = contact_forces_z[:, 0] > contact_forces_z[:, 1] * 2

    swing_foot_force = torch.where(
        left_is_support,
        right_foot_force,
        left_foot_force
    )

    drag_penalty = torch.clamp(swing_foot_force - drag_threshold, min=0.0)

    no_foot_contact = num_feet_in_contact == 0

    both_feet_drag_penalty = (
        torch.clamp(left_foot_force - drag_threshold, min=0.0) +
        torch.clamp(right_foot_force - drag_threshold, min=0.0)
    )

    reward = torch.where(
        single_foot_support,
        drag_penalty,
        torch.where(
            no_foot_contact,
            both_feet_drag_penalty,
            torch.zeros_like(drag_penalty)
        )
    )
    reward = torch.clamp(reward, max=threshold)

    return reward

def penalize_feet_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    swing_force_ratio: float = 0.15,
    support_force_ratio: float = 0.5,
    single_support_target: float = 0.11,
    single_support_scale: float = 5.0,
) -> torch.Tensor:
    """Penalize insufficient swing foot height during single foot support."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True).squeeze(-1)
    body_weight = total_mass * 9.8

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]

    swing_threshold = body_weight * swing_force_ratio
    support_threshold = body_weight * support_force_ratio

    left_force = contact_forces_z[:, 0]
    right_force = contact_forces_z[:, 1]

    left_is_swing = (left_force < swing_threshold) & (right_force > support_threshold)
    right_is_swing = (right_force < swing_threshold) & (left_force > support_threshold)
    single_foot_support = left_is_swing | right_is_swing

    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    swing_foot_height = torch.where(
        left_is_swing,
        feet_pos_z[:, 0],
        torch.where(
            right_is_swing,
            feet_pos_z[:, 1],
            torch.zeros_like(feet_pos_z[:, 0]),
        )
    )

    height_penalty = torch.clamp((single_support_target - swing_foot_height) * single_support_scale, min=0.0)

    height_reward = torch.where(
        single_foot_support,
        height_penalty,
        torch.zeros_like(height_penalty),
    )

    return height_reward

def twisted_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shank_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    horizontal_threshold: float = 0.3,
    shank_upright_threshold: float = 0.7,
) -> torch.Tensor:
    """Penalize twisted foot landing (contact force not perpendicular to foot surface)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces_w = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]

    shank_quat_w = asset.data.body_quat_w[:, shank_cfg.body_ids, :]

    num_envs = contact_forces_w.shape[0]
    num_shanks = shank_quat_w.shape[1]
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_envs * num_shanks, 3)

    shank_quat_flat = shank_quat_w.reshape(-1, 4)
    local_z = quat_rotate_inverse(shank_quat_flat, world_z)
    local_z = local_z.reshape(num_envs, num_shanks, 3)

    shank_uprightness = local_z[:, :, 2]

    is_shank_upright = shank_uprightness > shank_upright_threshold

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_feet = contact_forces_w.shape[1]

    contact_forces_w_flat = contact_forces_w.reshape(-1, 3)
    foot_quat_w_flat = foot_quat_w.reshape(-1, 4)

    contact_forces_local_flat = quat_rotate_inverse(foot_quat_w_flat, contact_forces_w_flat)

    contact_forces_local = contact_forces_local_flat.reshape(num_envs, num_feet, 3)

    force_local_x = contact_forces_local[:, :, 0]
    force_local_y = contact_forces_local[:, :, 1]
    force_local_z = contact_forces_local[:, :, 2]

    force_horizontal = torch.sqrt(force_local_x**2 + force_local_y**2)

    force_total = torch.norm(contact_forces_local, dim=-1)

    horizontal_ratio = force_horizontal / (force_total + 1e-6)

    contact_mag = torch.norm(contact_forces_w, dim=-1)
    is_in_contact = contact_mag > contact_threshold

    penalty_per_foot = torch.clamp(horizontal_ratio - horizontal_threshold, min=0.0)

    penalty_per_foot = torch.where(
        is_in_contact & is_shank_upright,
        penalty_per_foot,
        torch.zeros_like(penalty_per_foot)
    )

    total_penalty = torch.sum(penalty_per_foot, dim=1)

    return total_penalty