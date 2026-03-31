# from __future__ import annotations

# import math
# import numpy as np
# import os
# import torch
# from collections.abc import Sequence
# from dataclasses import MISSING
# from typing import TYPE_CHECKING

# from isaaclab.assets import Articulation
# from isaaclab.managers import CommandTerm, CommandTermCfg
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import FRAME_MARKER_CFG
# from isaaclab.utils import configclass
# from isaaclab.utils.math import (
#     quat_apply,
#     quat_error_magnitude,
#     quat_from_euler_xyz,
#     quat_inv,
#     quat_mul,
#     sample_uniform,
#     yaw_quat,
# )
# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv



# @configclass
# class MotionCommandCfg(CommandTermCfg):
#     """Configuration for the motion command."""

#     class_type: type = MotionCommand

#     asset_name: str = MISSING

#     motion_file: str = MISSING
#     anchor_body: str = MISSING
#     body_names: list[str] = MISSING

#     pose_range: dict[str, tuple[float, float]] = {}
#     velocity_range: dict[str, tuple[float, float]] = {}

#     joint_position_range: tuple[float, float] = (-0.52, 0.52)

#     adaptive_kernel_size: int = 3
#     adaptive_lambda: float = 0.8
#     adaptive_uniform_ratio: float = 0.1
#     adaptive_alpha: float = 0.001
    
#     start_hold_steps: int = 50
#     end_hold_steps: int = 50
    
#     anchor_pos_threshold: float = 0.25
#     anchor_ori_threshold: float = 0.3

#     anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
#     anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

#     body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
#     body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
