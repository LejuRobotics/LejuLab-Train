from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.managers.manager_term_cfg import RewardTermCfg


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_clip(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power (torque * velocity) on the articulation using L1 norm.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying the robot and joints. If None, defaults to "robot".
        joint_names: Joint names to specify joints directly. Can be a list of strings or a single string (regex pattern).
                    If provided, will override joint_names in asset_cfg. If both are None, uses all joints.

    Returns:
        Sum of absolute joint power (torque * velocity) for the specified joints.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)


class action_smoothness_l2(ManagerTermBase):
    """Penalize the second-order rate of change of actions using L2 squared kernel.
    
    This reward term penalizes action jerk (second derivative of actions) to encourage
    smooth control. It requires maintaining the previous-previous action state, which
    is not provided by Isaac Lab's action_manager (only prev_action is available).
    
    Therefore, this must be implemented as a class to maintain state across steps.
    """
    
    def __init__(self, cfg: "RewardTermCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = None

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Compute action smoothness penalty (second-order difference).
        
        Args:
            env: The environment instance.
            
        Returns:
            Sum of squared second-order action differences.
        """
        # Initialize on first call
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        
        # Compute second-order difference: action - 2*prev_action + prev_prev_action
        action_smoothness_l2 = torch.sum(
            torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            ),
            dim=1,
        )
        
        # Update state for next step
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    return torch.square(base_height - target_height)


# def track_lin_vel_xy_yaw_frame_exp(
#     env,
#     std: float,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_rotate_inverse(
#         yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3]
#     )
#     lin_vel_error = torch.sum(
#         torch.square(
#             env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
#         ),
#         dim=1,
#     )
#     return torch.exp(-lin_vel_error / std**2)


# def track_ang_vel_z_world_exp(
#     env,
#     command_name: str,
#     std: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(
#         env.command_manager.get_command(command_name)[:, 2]
#         - asset.data.root_com_ang_vel_w[:, 2]
#     )
#     return torch.exp(-ang_vel_error / std**2)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, violation_max: float = torch.inf) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0, max=violation_max), dim=1)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos_nominal
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff_angle = current_joint_pos - default_joint_pos
    reward = torch.sum(torch.abs(diff_angle), dim=-1)
    command = env.command_manager.get_command(command_name)
    zero_flag = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) < 0.05
    reward *= zero_flag
    return reward


# def joint_deviation_waist_l1(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize waist joint deviation from default position using L1 norm."""
#     asset: Articulation = env.scene[asset_cfg.name]
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward for lifting feet to target height during swing phase."""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def body_lin_acc_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies in z-axis direction using L2-kernel.
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying the robot and bodies.
        
    Returns:
        Sum of squared z-axis linear accelerations for all specified bodies.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2]), dim=1)


def feet_heading_alignment_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for aligning foot heading with command heading using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    command_term = env.command_manager.get_term(command_name)
    heading_angle = getattr(command_term, "heading_target")

    command_direction_xy = torch.stack([
        torch.cos(heading_angle),
        torch.sin(heading_angle)
    ], dim=1)

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs = foot_quat_w.shape[0]
    num_feet = foot_quat_w.shape[1]
    
    ang_vel_command = env.command_manager.get_command(command_name)[:, 2]

    heading_error = torch.abs(heading_angle - asset.data.heading_w)
    heading_error = torch.minimum(heading_error, 2 * torch.pi - heading_error)
    is_aligned = heading_error < 0.1
    
    is_turning_left = (ang_vel_command > 0.01) & (~is_aligned)
    is_turning_right = (ang_vel_command < -0.01) & (~is_aligned)

    is_left_foot = torch.zeros(num_feet, dtype=torch.bool, device=env.device)
    is_left_foot[0] = True
    
    foot_local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(num_envs * num_feet, 3)
    
    foot_quat_flat = foot_quat_w.reshape(-1, 4)
    foot_x_world = quat_rotate(foot_quat_flat, foot_local_x)
    foot_x_world = foot_x_world.reshape(num_envs, num_feet, 3)
    
    foot_x_xy = foot_x_world[:, :, :2]
    foot_x_xy_norm = torch.norm(foot_x_xy, dim=2, keepdim=True)
    foot_x_xy_norm = torch.clamp(foot_x_xy_norm, min=1e-6)
    foot_x_xy_normalized = foot_x_xy / foot_x_xy_norm

    command_direction_xy_expanded = command_direction_xy.unsqueeze(1).expand(-1, num_feet, -1)

    alignment = torch.sum(foot_x_xy_normalized * command_direction_xy_expanded, dim=2)

    alignment_adjusted = alignment.clone()

    is_turning_left_expanded = is_turning_left.unsqueeze(1).expand(-1, num_feet)
    is_turning_right_expanded = is_turning_right.unsqueeze(1).expand(-1, num_feet)
    is_left_foot_expanded = is_left_foot.unsqueeze(0).expand(num_envs, -1)

    left_turn_mask = is_turning_left_expanded & (~is_left_foot_expanded)
    alignment_adjusted[left_turn_mask] = -alignment[left_turn_mask]

    right_turn_mask = is_turning_right_expanded & is_left_foot_expanded
    alignment_adjusted[right_turn_mask] = -alignment[right_turn_mask]

    alignment_error = 1.0 - alignment_adjusted

    reward_per_foot = torch.exp(-alignment_error / (std ** 2))

    reward = torch.mean(reward_per_foot, dim=1)
    
    return reward


def body_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 0.2,
    max_dist: float = 0.5,
) -> torch.Tensor:
    """Reward based on distance between two bodies. Penalize bodies too close or too far."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    if len(asset_cfg.body_ids) != 2:
        raise ValueError(f"body_distance requires exactly 2 bodies, but got {len(asset_cfg.body_ids)}")
    
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    body_dist = torch.norm(body_pos[:, 0, :] - body_pos[:, 1, :], dim=1)
    
    d_min = torch.clamp(body_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(body_dist - max_dist, 0.0, 0.5)
    
    reward = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2.0
    
    return reward

def joint_deviation_l1_no_yaw_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation only when yaw command is absent or near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(torch.abs(diff_angle), dim=-1)
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] > 2:
        mask = torch.abs(cmd[:, 2]) < yaw_threshold
    else:
        mask = torch.ones(cmd.shape[0], device=reward.device, dtype=torch.bool)
    reward *= mask
    return reward


def joint_power_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)
