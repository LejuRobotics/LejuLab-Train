# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .rough_env_cfg import RobanS14RoughEnvCfg
from leju_robot.tasks.locomotion.velocity.config.robanS14.robanS14 import RobanS14_CFG


@configclass
class RobanS14FlatEnvCfg(RobanS14RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.Feet_L_scanner = None
        self.scene.Feet_R_scanner = None
        self.observations.critic.height_scan = None
        # no feet heights (requires Feet_L_scanner and Feet_R_scanner)
        self.observations.critic.feet_heights = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
class RobanS14FlatEnvCfg_PLAY(RobanS14FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1e9
        
        # disable randomization for play (make environment deterministic)
        # disable observation noise
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        
        # disable startup randomization events (make robot properties deterministic)
        self.events.randomize_rigid_body_com = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_link_mass = None
        self.events.add_joint_default_pos.params ={
            "asset_cfg": RobanS14_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.scale_joint_parameters = None
        
        self.events.reset_robot_joints = None
        
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-0, 0)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = True

