
from isaaclab.envs.mdp import *  # noqa: F401, F403
from leju_robot.tasks.amp.mdp import *
from leju_robot.tasks.amp.envs import ManagerBasedRLAMPEnv
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_state_or_walking(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Return command_state when available, otherwise assume walking mode (0)."""
    command_term = env.command_manager.get_term(command_name)
    cmd_state = getattr(command_term, "command_state", None)
    if cmd_state is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
    if cmd_state.ndim == 1:
        return cmd_state.unsqueeze(1).to(dtype=torch.long)
    return cmd_state.to(dtype=torch.long)


def rel_key_body_pos_b(env: ManagerBasedRLAMPEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject = env.scene[asset_cfg.name]
    key_body_positions = asset.data.body_pos_w[:, env.key_body_indexes]
    root_positions = asset.data.body_pos_w[:, env.ref_body_index]
    q = asset.data.root_quat_w
    return math_utils.quat_rotate_inverse(q.unsqueeze(-2), key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1)


def command_state(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _command_state_or_walking(env, command_name)


def ref_dof_pos_error(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    if hasattr(env, 'ref_dof_pos'):
        diff = asset.data.joint_pos - env.ref_dof_pos
        command_state = _command_state_or_walking(env, "base_velocity")
        non_walking_mask = (command_state[:, 0] != 0).unsqueeze(1).expand(-1, 21)

        diff = diff * non_walking_mask

        return diff
    else:
        return torch.zeros((env.num_envs, 21), device=env.device)



