from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointAction, JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ext_roban.tasks.manager_based.amp.mdp import actions_cfg





class MyJointPositionToLimitsAction(JointPositionAction):

    cfg: actions_cfg.MyJointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.MyJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        if cfg.rescale_to_limits:
            dof_lower_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
            dof_upper_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
            self._scale = dof_upper_limits - dof_lower_limits
