from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of horizontal foot speeds weighted by contact flags (slide magnitude when feet touch)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_slide_yaw(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Yaw-axis foot angular velocity when in contact, gated by nonzero yaw command."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, sensor_cfg.body_ids, 2:3]
    reward_ang_vel = torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    commands = env.command_manager.get_command(command_name)
    has_rotation_cmd = torch.abs(commands[:, 2]) > 0.1
    return reward_ang_vel * has_rotation_cmd.float()


def track_lin_vel_xy_yaw_frame_piecewise_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Piecewise yaw-frame horizontal velocity tracking reward.

    - ``lateral_only``: when ``|vx_cmd| < 0.2`` and ``|vy_cmd| > 0.1``, reward ``vy`` tracking
      and suppress measured ``vx``.
    - ``forward_only``: when ``|vx_cmd| > 0.1`` and ``|vy_cmd| < 0.2``, reward ``vx`` tracking
      and suppress measured ``vy``.
    - ``full_xy``: when ``|vx_cmd| > 0.2`` and ``|vy_cmd| > 0.2``, reward full ``x-y`` tracking.

    The small overlap region ``0.1 < |vx_cmd| < 0.2`` and ``0.1 < |vy_cmd| < 0.2`` satisfies
    both ``lateral_only`` and ``forward_only``; because ``forward_only`` is applied later, it takes
    priority there. Commands outside these regimes keep zero error and therefore reward ``1``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    vel_yaw = quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])

    vx_cmd = command[:, 0]
    vy_cmd = command[:, 1]
    vx = vel_yaw[:, 0]
    vy = vel_yaw[:, 1]

    lateral_only = (torch.abs(vx_cmd) < 0.25) & (torch.abs(vy_cmd) > 0.05)
    forward_only = (torch.abs(vx_cmd) > 0.05) & (torch.abs(vy_cmd) < 0.25)
    full_xy = (torch.abs(vx_cmd) > 0.25) & (torch.abs(vy_cmd) > 0.25)

    lin_vel_error = torch.zeros(env.num_envs, device=env.device)
    lin_vel_error = torch.where(lateral_only, torch.square(vy_cmd - vy) + torch.square(vx), lin_vel_error)
    lin_vel_error = torch.where(forward_only, torch.square(vx_cmd - vx) + torch.square(vy), lin_vel_error)
    lin_vel_error = torch.where(full_xy, torch.square(vx_cmd - vx) + torch.square(vy_cmd - vy), lin_vel_error)

    reward = torch.exp(-lin_vel_error / std**2)
    return reward 


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Binary indicator (float): any monitored body exceeds contact force threshold."""
    _ = command_name
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    all_net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    all_contact_norms = torch.norm(all_net_contact_forces, dim=-1)
    all_is_contact = (all_contact_norms > threshold).any(dim=1)
    return all_is_contact.float()


class action_smoothness_l2(ManagerTermBase):
    """L2 penalty on discrete action curvature (second difference across steps)."""

    def __init__(self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        super().__init__(env, cfg)
        self.prev_prev_action = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear lagged actions for reset environments."""
        if self.prev_prev_action is None:
            return
        self.prev_prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
    ):
        """Return sum of squared second differences of actions; updates lag buffer."""
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        action_diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + self.prev_prev_action)
        term = torch.sum(action_diff, dim=1)
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return term


def joint_deviation_l1_straight_only(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    angular_threshold: float = 0.1,
    linear_threshold: float = 0.1,
) -> torch.Tensor:
    """L1 deviation from default pose only when commanded motion is roughly straight."""
    command = env.command_manager.get_command(command_name)
    ang_vel_command = command[:, 2]
    is_straight = (torch.abs(ang_vel_command) < angular_threshold) & (torch.abs(command[:, 1]) < linear_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - joint_pos_default), dim=1)
    return torch.where(is_straight, deviation, torch.zeros_like(deviation))


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Sum of absolute joint deviations from default pose (``command_name`` reserved for API symmetry)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - joint_pos_default), dim=1)
    return deviation

class joint_mean_acc_l2_mode(ManagerTermBase):
    """Per-step joint acceleration penalty with clipping above a threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self.prev_joint_vel = asset.data.joint_vel.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """No-op reset hook (velocity buffer continuity handled implicitly)."""
        _ = env_ids
        asset_cfg: SceneEntityCfg = self.cfg.params["asset_cfg"]
        _asset: RigidObject | Articulation = self._env.scene[asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = 8000.0,
        command_name: str = "base_velocity",
    ):
        """Sum of clipped squared joint accelerations minus ``threshold``."""
        asset: Articulation = env.scene[asset_cfg.name]
        if asset_cfg.joint_ids == slice(None):
            joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)
        elif isinstance(asset_cfg.joint_ids, slice):
            joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)[asset_cfg.joint_ids]
        else:
            joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)
        square_acc = torch.square((asset.data.joint_vel[:, joint_ids] - self.prev_joint_vel[:, joint_ids]) / env.step_dt)
        square_acc = torch.clip(square_acc - threshold, min=0.0, max=7500)
        _ = command_name
        acc_l2 = torch.sum(square_acc, dim=1)
        self.prev_joint_vel = asset.data.joint_vel.clone()
        return acc_l2


def contact_forces_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    violation_max: float = torch.inf,
    violation_min: float = 0.0,
) -> torch.Tensor:
    """Penalty for peak net contact force above ``threshold`` on bodies currently in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    violation *= in_contact
    return torch.sum(violation.clip(min=violation_min, max=violation_max), dim=1)


def feet_aligned_support_penalty_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
    min_fore_aft_separation_base: float = 0.02,
    min_fore_aft_separation_per_vx: float = 0.5,
    min_sep_max: float | None = 0.30,
    command_ref: float = 0.8,
) -> torch.Tensor:
    """Penalty for double support with feet too close in fore-aft, measured in the yaw frame.

    The foot relative positions are rotated by the inverse
    of the root yaw-only quaternion, so fore-aft separation is measured in a true yaw-aligned
    horizontal frame and is not affected by roll/pitch.
    """
    command = env.command_manager.get_command(command_name)
    vx_cmd = command[:, 0]
    cmd_mag = torch.abs(vx_cmd)
    cmd_scale = torch.clamp(cmd_mag / max(command_ref, 1e-6), max=1.0)
    moving_fwd = cmd_mag > 0.05
    cmd_gate = cmd_scale * moving_fwd.float()

    min_sep_req = min_fore_aft_separation_base + min_fore_aft_separation_per_vx * cmd_mag
    if min_sep_max is not None:
        min_sep_req = torch.clamp(min_sep_req, max=min_sep_max)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = hist.norm(dim=-1).max(dim=1)[0] > 1.0
    both_feet = torch.all(in_contact, dim=1)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos = asset.data.root_pos_w
    root_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    rel_w = foot_pos_w - root_pos.unsqueeze(1)
    n = env.num_envs
    n_feet = foot_pos_w.shape[1]
    q_exp = root_yaw_quat.unsqueeze(1).expand(-1, n_feet, -1).reshape(n * n_feet, 4)
    rel_flat = rel_w.reshape(n * n_feet, 3)
    foot_yaw = math_utils.quat_rotate_inverse(q_exp, rel_flat).view(n, n_feet, 3)
    fore_aft_sep = torch.abs(foot_yaw[:, 0, 0] - foot_yaw[:, 1, 0])
    shortfall = torch.clamp(min_sep_req - fore_aft_sep, min=0.0)
    align_cost = torch.square(shortfall)
    not_turning = torch.abs(command[:, 2]) < 0.1

    return both_feet.float() * align_cost * cmd_gate * not_turning


def both_feet_grounded(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.3,
    max_grounded_time: float = 0.125,
) -> torch.Tensor:
    """Penalize prolonged double support while commanding locomotion (optional standing-env exemption)."""
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    # Isaac Lab UniformVelocityCommand / NormalVelocityCommand: bool mask for rel_standing_envs.
    is_standing_env = getattr(command_term, "is_standing_env", None)
    is_moving = (
        (torch.abs(command[:, 0]) > velocity_threshold)
        | (torch.abs(command[:, 1]) > velocity_threshold)
        | (torch.abs(command[:, 2]) > velocity_threshold)
    )
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = contact_forces_hist.norm(dim=-1).max(dim=1)[0] > 1.0
    both_feet_grounded = torch.all(in_contact, dim=1)
    if not hasattr(env, "_both_feet_grounded_time"):
        env._both_feet_grounded_time = torch.zeros(env.num_envs, device=env.device)
    env._both_feet_grounded_time = torch.where(
        both_feet_grounded,
        env._both_feet_grounded_time + env.step_dt,
        torch.zeros_like(env._both_feet_grounded_time),
    )
    if is_standing_env is not None:
        env._both_feet_grounded_time = torch.where(
            is_standing_env,
            torch.zeros_like(env._both_feet_grounded_time),
            env._both_feet_grounded_time,
        )
    penalty = torch.clamp(env._both_feet_grounded_time - max_grounded_time, min=0.0)
    apply_penalty = is_moving
    if is_standing_env is not None:
        apply_penalty = apply_penalty & ~is_standing_env
    return torch.where(apply_penalty, penalty * 2.0, torch.zeros_like(penalty))


def foot_clearance_reward_floor_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    gate_floor: float = 0.02,
) -> torch.Tensor:
    """Foot clearance reward with a lower-bounded foot-speed gate.

    ``gate = gate_floor + (1 - gate_floor) * tanh(tanh_mult * ||v_xy||)`` per foot, so at zero
    horizontal foot speed the height error is still weighted by ``gate_floor`` instead of being fully masked out.

    Args:
        gate_floor: In ``[0, 1)``. Typical small values: ``0.02`` … ``0.15``.
    """
    gate_floor = max(0.0, min(float(gate_floor), 0.999))
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    foot_velocity_tanh = torch.tanh(tanh_mult * speed)
    gate = gate_floor + (1.0 - gate_floor) * foot_velocity_tanh
    reward = foot_z_target_error * gate
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_gait_swing(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """Reward swing-phase agreement between phased gait schedule and foot contact."""
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
        # Reference phase says stance when phase fraction < threshold.
        is_stance = leg_phase[:, i] < threshold
        # XNOR: reward when stance prediction matches contact; multiply by swing mask (~is_stance).
        reward += (~(is_stance ^ is_contact[:, i])).float() * (~is_stance)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def _arm_swing_error_to_reward_piecewise(
    e: torch.Tensor,
    max_reward: float,
    err_first_segment: float,
) -> torch.Tensor:
    """Map mean absolute error (rad) to ``[0, max_reward]``.

    - ``e in [0, err_first_segment]``: steeper linear drop (ends at ``0.2 * max_reward``).
    - ``e in (err_first_segment, 1]``: shallower linear drop to 0 at ``e == 1``.
    - ``e > 1`` (after clamp): 0.
    """
    e = e.clamp(0.0, 1.0)
    a = e <= err_first_segment
    b = e > err_first_segment
    r = torch.zeros_like(e)
    r[a] = max_reward * (1.0 - 0.8 * (e[a] / err_first_segment))
    r[b] = (0.2 * max_reward) * (1.0 - e[b]) / (1.0 - err_first_segment)
    return r.clamp(0.0, max_reward)


def arm_swing_gait_phase_reward(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    same_side_target: float = 1.0,
    err_piecewise_split: float = 0.5,
    command_name: str | None = None,
    leg_is_left: list[bool] | None = None,
    elbow_default_offset: float = 0.5,
) -> torch.Tensor:
    """Arm swing reward using the same gait phase as :func:`feet_gait_swing`; does not read foot contacts.

    ``global_phase = ((episode_length * step_dt) % period) / period``, then each foot adds
    ``offset`` and the same ``threshold`` splits stance vs swing. During **expected swing**:

    - ``zarm_*1``: same-side / opposite shoulder pitch track ``same_side_target`` / ``-same_side_target``.
    - ``zarm_*4`` (elbow contralateral to foot ``i``): tracks ``default_joint_pos - elbow_default_offset`` rad.

    Shoulder and elbow sub-rewards each map through the same piecewise curve to ``[0, 1]``, then average;
    per-environment values take the **max over legs** and are **clamped to ``[0, 1]``**.

    ``sensor_cfg`` is only used so ``len(body_ids)`` matches ``offset`` length; contact is not used.
    """
    n_foot = len(sensor_cfg.body_ids)
    if len(offset) != n_foot:
        raise ValueError("arm_swing_gait_phase_reward: len(offset) must match sensor_cfg.body_ids count")

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    if leg_is_left is None:
        if n_foot != 2:
            raise ValueError("arm_swing_gait_phase_reward: set leg_is_left when foot count != 2")
        leg_is_left = [True, False]
    if len(leg_is_left) != n_foot:
        raise ValueError("arm_swing_gait_phase_reward: leg_is_left length must match feet")

    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    joint_def = asset.data.default_joint_pos
    zl = asset.find_joints("zarm_l1_joint")[0]
    zr = asset.find_joints("zarm_r1_joint")[0]
    zl4 = asset.find_joints("zarm_l4_joint")[0]
    zr4 = asset.find_joints("zarm_r4_joint")[0]
    q_l1 = q[:, zl].reshape(env.num_envs)
    q_r1 = q[:, zr].reshape(env.num_envs)

    out = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(n_foot):
        is_stance = leg_phase[:, i] < threshold
        is_swing = ~is_stance
        if leg_is_left[i]:
            t_l, t_r = same_side_target, -same_side_target
            q_el = q[:, zr4].reshape(env.num_envs)
            t_el = joint_def[:, zr4].reshape(env.num_envs) - elbow_default_offset
        else:
            t_l, t_r = -same_side_target, same_side_target
            q_el = q[:, zl4].reshape(env.num_envs)
            t_el = joint_def[:, zl4].reshape(env.num_envs) - elbow_default_offset
        e_sh = 0.5 * (torch.abs(q_l1 - t_l) + torch.abs(q_r1 - t_r))
        r_sh = _arm_swing_error_to_reward_piecewise(
            e_sh, max_reward=1.0, err_first_segment=err_piecewise_split
        )
        e_el = torch.abs(q_el - t_el)
        r_el = _arm_swing_error_to_reward_piecewise(
            e_el, max_reward=1.0, err_first_segment=err_piecewise_split
        )
        r_leg = 0.5 * (r_sh + r_el)
        out = torch.maximum(out, r_leg * is_swing.float())

    command = env.command_manager.get_command(command_name)
    if command_name is not None:
        cmd_norm = torch.norm(command, dim=1)
        out = out * (cmd_norm > 0.1).float()

    reward = torch.clamp(out, 0.0, 1.0).reshape(env.num_envs)
    # While turning (|yaw command| above threshold), scale arm swing reward up; matches cmd gate when command_name is set.
    if command_name is not None:
        turning = torch.abs(command[:, 2]) > 0.1
        reward = reward * (1.0 + 2.0*turning.float())
    return reward


def arm_swing_gait_phase_reward_elbow(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    same_side_target: float = 1.0,
    err_piecewise_split: float = 0.5,
    command_name: str | None = None,
    leg_is_left: list[bool] | None = None,
    elbow_default_offset: float = 0.5,
    turning_boost: float = 2.0,
) -> torch.Tensor:
    """Arm swing reward like :func:`arm_swing_gait_phase_reward`, with a phase-varying elbow target.

    Shoulders match the same piecewise tracking as the base reward. For the contralateral ``zarm_*4``
    (elbow) joint, the target is

    ``default_joint_pos - elbow_default_offset * abs(sin(pi * s))``,

    where ``s`` is the **contralateral** leg's swing progress in ``[0, 1]`` (0 at swing start/end,
    0.5 at mid-swing). When the contralateral leg is in stance, ``s = 0`` so the target equals the
    default pose. The elbow target reaches its minimum when the opposite leg is halfway through
    its swing phase.

    Exactly two feet are required so the contralateral leg is unambiguous. ``sensor_cfg`` is only
    used so ``len(body_ids)`` matches ``offset`` length; contact is not used.
    """
    n_foot = len(sensor_cfg.body_ids)
    if n_foot != 2:
        raise ValueError(
            "arm_swing_gait_phase_reward_elbow: requires exactly two feet (contralateral elbow)"
        )
    if len(offset) != n_foot:
        raise ValueError(
            "arm_swing_gait_phase_reward_elbow: len(offset) must match sensor_cfg.body_ids count"
        )

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    if leg_is_left is None:
        leg_is_left = [True, False]
    if len(leg_is_left) != n_foot:
        raise ValueError("arm_swing_gait_phase_reward_elbow: leg_is_left length must match feet")

    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    joint_def = asset.data.default_joint_pos
    zl = asset.find_joints("zarm_l1_joint")[0]
    zr = asset.find_joints("zarm_r1_joint")[0]
    zl4 = asset.find_joints("zarm_l4_joint")[0]
    zr4 = asset.find_joints("zarm_r4_joint")[0]
    q_l1 = q[:, zl].reshape(env.num_envs)
    q_r1 = q[:, zr].reshape(env.num_envs)

    swing_span = max(1.0 - threshold, 1e-6)

    out = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(n_foot):
        is_stance = leg_phase[:, i] < threshold
        is_swing = ~is_stance
        j = 1 - i
        phi_j = leg_phase[:, j]
        in_swing_j = phi_j >= threshold
        s = torch.where(
            in_swing_j,
            (phi_j - threshold) / swing_span,
            torch.zeros_like(phi_j),
        )
        sin_term = torch.abs(torch.sin(math.pi * s))

        if leg_is_left[i]:
            t_l, t_r = same_side_target, -same_side_target
            q_el = q[:, zr4].reshape(env.num_envs)
            jdef_el = joint_def[:, zr4].reshape(env.num_envs)
        else:
            t_l, t_r = -same_side_target, same_side_target
            q_el = q[:, zl4].reshape(env.num_envs)
            jdef_el = joint_def[:, zl4].reshape(env.num_envs)

        t_el = jdef_el - elbow_default_offset * sin_term
        e_sh = 0.5 * (torch.abs(q_l1 - t_l) + torch.abs(q_r1 - t_r))
        r_sh = _arm_swing_error_to_reward_piecewise(
            e_sh, max_reward=1.0, err_first_segment=err_piecewise_split
        )
        e_el = torch.abs(q_el - t_el)
        r_el = _arm_swing_error_to_reward_piecewise(
            e_el, max_reward=1.0, err_first_segment=err_piecewise_split
        )
        r_leg = 0.5 * (r_sh + r_el)
        out = torch.maximum(out, r_leg * is_swing.float())

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        cmd_norm = torch.norm(command, dim=1)
        out = out * (cmd_norm > 0.1).float()

    reward = torch.clamp(out, 0.0, 1.0).reshape(env.num_envs)
    if command_name is not None:
        turning = torch.abs(command[:, 2]) > 0.1
        reward = reward * (1.0 + (turning_boost - 1.0) * turning.float())
    return reward


class stand_still_without_cmd_last(ManagerTermBase):
    """L1 joint deviation when current and previous commands are near idle (standing stabilization)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_command: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Align lagged command buffer with current commands after episode reset."""
        if self.prev_command is None:
            return
        command_name = self.cfg.params["command_name"]
        rl_env = cast("ManagerBasedRLEnv", self._env)
        current_command = rl_env.command_manager.get_command(command_name)
        if env_ids is None:
            self.prev_command[:] = current_command
        else:
            self.prev_command[env_ids] = current_command[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Return deviation penalty only when consecutive commands stay small and external forces are low."""
        asset: Articulation = env.scene[asset_cfg.name]
        diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

        reward = torch.sum(torch.abs(diff_angle), dim=-1)

        current_command = env.command_manager.get_command(command_name)
        if self.prev_command is None:
            self.prev_command = current_command.clone()
        assert self.prev_command is not None
        prev_command = self.prev_command

        reward *= torch.norm(current_command[:, :3], dim=1) < 0.1
        reward *= current_command[:, 2] < 0.1
        reward *= torch.norm(prev_command[:, :3], dim=1) < 0.1
        reward *= prev_command[:, 2] < 0.1
        reward *= torch.norm(asset._external_force_b[:, 0, :], dim=1) < 20

        self.prev_command = current_command.clone()
        return reward

def turn_swing_knee_flex_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
    right_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["leg_r4_joint"]),
    left_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["leg_l4_joint"]),
    command_name: str = "base_velocity",
    flex_knots: Sequence[float] = (0.2, 0.7, 1.0, 1.3),
    yaw_cmd_deadband: float = 0.1,
    linear_cmd_deadband: float = 0.1,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Positive cost for insufficient flex of the swing-side knee during turn-in-place.

    This term is active only when commanded motion is near pure rotation:
    ``|vx_cmd| <= linear_cmd_deadband``, ``|vy_cmd| <= linear_cmd_deadband``,
    and ``|wz_cmd| > yaw_cmd_deadband``.

    The knee score shape follows ``flex_knots=[a0, a1, a2, a3]``:
    - ``q < a0``: ``(q - a0)`` (negative)
    - ``a0 <= q <= a1``: linear 0 -> 1
    - ``a1 < q < a2``: 1
    - ``a2 <= q <= a3``: linear 1 -> 0
    - ``q > a3``: 0

    Only knees belonging to feet that are currently not in contact (swing feet)
    are penalized.
    """
    knots = list(flex_knots)
    if len(knots) != 4:
        raise ValueError(f"flex_knots must have length 4, got {len(knots)}")
    a0, a1, a2, a3 = (float(knots[0]), float(knots[1]), float(knots[2]), float(knots[3]))
    if not (a0 < a1 < a2 < a3):
        raise ValueError(f"flex_knots must be strictly increasing, got {knots}")

    command = env.command_manager.get_command(command_name)
    cmd_vx = command[:, 0]
    cmd_vy = command[:, 1]
    cmd_wz = command[:, 2]
    is_turn_in_place = (
        (torch.abs(cmd_vx) <= linear_cmd_deadband)
        & (torch.abs(cmd_vy) <= linear_cmd_deadband)
        & (torch.abs(cmd_wz) > yaw_cmd_deadband)
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = net_forces_hist.norm(dim=-1).max(dim=1)[0] > contact_force_threshold
    is_swing = ~is_contact

    # If regex order is [left, right], reorder to [right, left] to match knee terms.
    if is_swing.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 foot bodies for swing mask, got {is_swing.shape[1]}")
    swing_right = is_swing[:, 1]
    swing_left = is_swing[:, 0]

    asset: Articulation = env.scene[right_knee_cfg.name]
    q_r = asset.data.joint_pos[:, right_knee_cfg.joint_ids].reshape(env.num_envs)
    q_l = asset.data.joint_pos[:, left_knee_cfg.joint_ids].reshape(env.num_envs)

    band01 = max(a1 - a0, 1e-6)
    band23 = max(a3 - a2, 1e-6)

    def _score(q: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(q)
        penalty = torch.where(q < a0, q - a0, z)
        rise = torch.where((q >= a0) & (q <= a1), (q - a0) / band01, z)
        plateau = torch.where((q > a1) & (q < a2), torch.ones_like(q), z)
        decay = torch.where((q >= a2) & (q <= a3), (a3 - q) / band23, z)
        return penalty + rise + plateau + decay

    peak = 1.0
    gap_r = torch.clamp(peak - _score(q_r), min=0.0) * swing_right.float()
    gap_l = torch.clamp(peak - _score(q_l), min=0.0) * swing_left.float()
    penalty = gap_r + gap_l
    return torch.where(is_turn_in_place, penalty, torch.zeros_like(penalty))


def turn_in_place_leg1_abs_penalty(
    env: ManagerBasedRLEnv,
    left_leg1_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["leg_l1_joint"]),
    right_leg1_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["leg_r1_joint"]),
    command_name: str = "base_velocity",
    yaw_cmd_deadband: float = 0.05,
    linear_cmd_deadband: float = 0.1,
    abs_thresh_low: float = 0.5,
    abs_thresh_high: float = 0.7,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Hip-abduction (*leg1*) magnitude cost during commanded turn-in-place.

    Active only when ``|vx_cmd|,|vy_cmd| <= linear_cmd_deadband`` and
    ``|wz_cmd| > yaw_cmd_deadband``.

    Let ``x = max(|q_leg_l1|, |q_leg_r1|)``. Cost is ``0`` for ``x <= abs_thresh_low``,
    linear from ``0`` to ``max_penalty`` when ``abs_thresh_low < x <= abs_thresh_high``,
    and ``max_penalty`` when ``x > abs_thresh_high``.

    ``left_leg1_cfg`` and ``right_leg1_cfg`` must refer to the **same** articulated asset
    (same ``name``); only joint name lists may differ.
    """
    if not (abs_thresh_low < abs_thresh_high):
        raise ValueError(f"require abs_thresh_low < abs_thresh_high, got {abs_thresh_low}, {abs_thresh_high}")

    command = env.command_manager.get_command(command_name)
    cmd_vx = command[:, 0]
    cmd_vy = command[:, 1]
    cmd_wz = command[:, 2]
    is_turn_in_place = (
        (torch.abs(cmd_vx) <= linear_cmd_deadband)
        & (torch.abs(cmd_vy) <= linear_cmd_deadband)
        & (torch.abs(cmd_wz) > yaw_cmd_deadband)
    )

    if left_leg1_cfg.name != right_leg1_cfg.name:
        raise ValueError(
            "turn_in_place_leg1_abs_penalty: left_leg1_cfg.name and right_leg1_cfg.name must match "
            f"(got {left_leg1_cfg.name!r} vs {right_leg1_cfg.name!r})."
        )
    asset: Articulation = env.scene[left_leg1_cfg.name]
    lid_ids = asset.find_joints("leg_l1_joint")[0]
    rid_ids = asset.find_joints("leg_r1_joint")[0]
    if len(lid_ids) > 1:
        lid_ids = lid_ids[:1]
    if len(rid_ids) > 1:
        rid_ids = rid_ids[:1]
    q_l = asset.data.joint_pos[:, lid_ids].reshape(env.num_envs)
    q_r = asset.data.joint_pos[:, rid_ids].reshape(env.num_envs)
    x = torch.maximum(torch.abs(q_l), torch.abs(q_r))

    band = max(abs_thresh_high - abs_thresh_low, 1e-6)
    in_ramp = (x > abs_thresh_low) & (x <= abs_thresh_high)
    above = x > abs_thresh_high
    ramp = (x - abs_thresh_low) / band * max_penalty
    cost = torch.where(
        above,
        torch.full_like(x, max_penalty),
        torch.where(in_ramp, ramp, torch.zeros_like(x)),
    )
    return torch.where(is_turn_in_place, cost, torch.zeros_like(cost))


def feet_y_distance_straight(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
) -> torch.Tensor:
    """Linear foot-spacing penalty around target y-distance during straight commands."""
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    lin_vel_x = command[:, 0]
    lin_vel_y = command[:, 1]
    ang_vel_z = command[:, 2]

    is_straight = (torch.abs(ang_vel_z) < 0.1) & (torch.abs(lin_vel_y) < 0.1) & (torch.abs(lin_vel_x) > 0.05)
    joint_ids = asset.find_bodies(asset_cfg.body_names)[0]
    feet_pos = asset.data.body_pos_w[:, joint_ids, :]
    leftfoot = feet_pos[:, 0] - asset.data.root_link_pos_w[:, :]
    rightfoot = feet_pos[:, 1] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = quat_rotate_inverse(asset.data.root_link_quat_w[:, :], leftfoot)
    rightfoot_b = quat_rotate_inverse(asset.data.root_link_quat_w[:, :], rightfoot)

    y_distance = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1])
    penalty = (
        torch.exp(5.0 * torch.clamp(0.20 - y_distance, min=0.0))
        - 1.0
        + torch.exp(5.0 * torch.clamp(y_distance - 0.23, min=0.0))
        - 1.0
    )
    # target_y_distance = 0.23
    # penalty = torch.abs(y_distance - target_y_distance)
    return penalty * is_straight.float()

def fft_dof_symmetry(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        joint_names_pairs: list = ["leg_[l,r]4_link"],
        linear_x_threshold: float = 0.4,  
        angular_threshold: float = 0.6,
        command_name: str = "base_velocity"
) -> torch.Tensor:
    
    """symmetry"""
    # extract the used quantities (to enable type-hinting)
    if not hasattr(env, 'joint_action_history_sym'):
            env.joint_action_history_sym = torch.zeros(
                env.num_envs, len(joint_names_pairs)*2, 100,
                device=env.device
            )
    env.joint_action_history_sym = torch.roll(env.joint_action_history_sym, 1,dims=2)

    asset: Articulation = env.scene[asset_cfg.name]

    symmetry_metric = torch.zeros(env.num_envs, device=env.device)
    for i,joint_pair in enumerate(joint_names_pairs):
        joint_idxs = asset.find_joints(joint_pair)[0]
    
    #fft
        joint_action = env.action_manager.action[:,env.action_manager._terms["joint_pos"]._joint_ids][:,joint_idxs]#action
        
        env.joint_action_history_sym[:, i*2:i*2+2, 0] = joint_action
        joint_history = env.joint_action_history_sym[:, i*2:i*2+2]
        joint_history_centered = joint_history - joint_history.mean(dim=2, keepdim=True)

        # N = joint_history_centered.shape[-1] 
        # fs = 1.0 / env.step_dt               
        # freqs = torch.fft.rfftfreq(N, d=1/fs)

        fft_vals = torch.fft.rfft(joint_history_centered, dim=2)
        fft_magnitudes = torch.abs(fft_vals)

        topk_vals, topk_idx = torch.topk(fft_magnitudes[:,0,:], k=20, dim=1)

        single_joint_diff = torch.sum(torch.abs(fft_magnitudes[:,0,topk_idx[0]] - fft_magnitudes[:,1,topk_idx[0]]),dim=-1)



        symmetry_metric += single_joint_diff
    without_external_force_apply = torch.norm(asset._external_force_b[:,0,:], dim=1)<5
    big_angular_and_linear = (torch.abs(env.command_manager.get_command(command_name)[:, 2])>angular_threshold) & (env.command_manager.get_command(command_name)[:, 0] > linear_x_threshold)
    symmetry_metric *= without_external_force_apply
    symmetry_metric[big_angular_and_linear] *= 0.3
    return torch.clip(symmetry_metric,max=300)