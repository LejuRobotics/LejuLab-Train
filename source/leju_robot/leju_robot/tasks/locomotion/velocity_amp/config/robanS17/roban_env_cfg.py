# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from leju_robot.tasks.locomotion.velocity_amp.velocity_amp_env_cfg import LocomotionVelocityAMPRoughEnvCfg

##
# Pre-defined configs
##
from leju_robot.assets.leju import RobanS17_AMP_CFG
from leju_robot.tasks.locomotion.velocity_amp.terrains import ROUGH_TERRAINS_CFG
from leju_robot.tasks.amp.utils import motions
MOTIONS_DIR = os.path.dirname(os.path.abspath(motions.__file__))


PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "waist_yaw_joint",
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
        "leg_l6_joint",
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_r6_joint",
        "zarm_l1_joint",
        "zarm_l2_joint",
        "zarm_l3_joint",
        "zarm_l4_joint",
        "zarm_r1_joint",
        "zarm_r2_joint",
        "zarm_r3_joint",
        "zarm_r4_joint",
    ],
    preserve_order=True
)


RobanS17_ACTION_SCALE = 0.25

@configclass
class RobanS2RoughEnvCfg(LocomotionVelocityAMPRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = RobanS17_AMP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # reduce action scale
        self.actions.joint_pos.scale = RobanS17_ACTION_SCALE
        self.actions.joint_pos.joint_names = PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names
        self.observations.policy.joint_pos.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.policy.joint_vel.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.key_body_names = ["zarm_r4_link", "zarm_l4_link", "leg_r6_link", "leg_l6_link"]
        self.reference_body = "base_link"


@configclass
class RobanS2RoughEnvCfg_PLAY(RobanS2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # remove random pushing event
        self.events.base_external_force_torque = None







@configclass
class RobanS2MixEnvCfg(RobanS2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.observations.critic.height_scan = None
        self.num_amp_observations = 5
        
        self.motion_mode={      
        # 参考数据默认比例均为1
        "bipeds17_0307_sleg":{
            
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","侧移_右_低速_Skeleton_sleg_retarget.npz",                                    ): 1.0,
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","侧移_右_高速_Skeleton_sleg_retarget.npz",                                    ): 1.0,
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","侧移_左_低速_Skeleton_sleg_retarget.npz",                                    ): 1.0,
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","侧移_左_高速_Skeleton_sleg_retarget.npz",                                    ): 1.0,

            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_中速_逆时针_Skeleton 001_arms_retarget.npz",                         ): 1.0, 
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_中速_顺时针_000_Skeleton 001_segment_000_f0-1729_arms_retarget.npz", ): 1.0,
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_低速_逆时针_Skeleton 001_arms_retarget.npz",                         ): 1.0, 
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_低速_顺时针_Skeleton 001_arms_retarget.npz",                         ): 1.0, 
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_加减速_逆时针_小摆手_000_Skeleton_arms_retarget.npz",                      ): 1.0,       
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_加减速_逆时针_小摆手_Skeleton_segment_000_f0-2218_arms_retarget.npz",      ): 1.0,        
            os.path.join(MOTIONS_DIR,"head","npz","原地转圈_加减速_顺时针_小摆手_Skeleton_arms_retarget.npz",                          ): 1.0,    

            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","后退_中速_小摆手_000_Skeleton_sleg_retarget.npz",                            ): 1.0,  
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","后退_低速_小摆手_000_Skeleton_segment_000_f0-1444_sleg_retarget.npz",        ): 1.0,    
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","后退_低速_小摆手_Skeleton_segment_000_f530-1949_sleg_retarget.npz",          ): 1.0,

            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","圆_Skeleton_sleg_retarget.npz",                                            ): 1.0,  
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","弧线_中右_Skeleton_segment_000_f247-1360_sleg_retarget.npz",                ): 1.0,
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","弧线_中左_Skeleton_segment_000_f210-1117_sleg_retarget.npz",                ): 1.0,       
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","弧线_大右_Skeleton_sleg_retarget.npz",                                      ): 1.0,       
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","弧线_大左_Skeleton_segment_000_f270-1064_sleg_retarget.npz",                ): 1.0,            
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","弧线_小右_Skeleton_sleg_retarget.npz",                                      ): 1.0,            
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","椭圆_Skeleton_segment_000_f0-4173_sleg_retarget.npz",                      ): 1.0,         
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","椭圆_Skeleton_segment_001_f4362-5127_sleg_retarget.npz",                   ): 1.0,     

            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","直行_中速_小摆手_003_Skeleton_segment_000_f0-981_sleg_retarget.npz",         ): 1.0,      
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","直行_中速_小摆手_005_Skeleton_segment_000_f195-1206_sleg_retarget.npz",      ): 1.0,         
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","直行_中速_小摆手_006_Skeleton_segment_000_f0-1304_sleg_retarget.npz",        ): 1.0,                
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","直行_中速_小摆手_Skeleton_segment_001_f558-1349_sleg_retarget.npz",          ): 1.0,           
            os.path.join(MOTIONS_DIR,"0919_sleg_v0307_roban17","npz","直行_低速_小摆手_Skeleton_segment_000_f0-1271_sleg_retarget.npz",            ): 1.0,                      
            
            }        
        }
        self.motion_file = self.motion_mode["bipeds17_0307_sleg"]

@configclass
class RobanS2MixEnvCfg_PLAY(RobanS2MixEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # Keep play config robust even if base_external_force_torque is disabled upstream.
        base_external_force_torque = getattr(self.events, "base_external_force_torque", None)
        if base_external_force_torque is not None:
            base_external_force_torque.params["probability"] = 1
            base_external_force_torque.params["force_range"]["x"] = (0, 0)

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.events.randomize_rigid_body_com = None
        self.events.physics_material = None
        # self.events.add_base_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_link_mass = None
        if self.curriculum is not None:
            self.curriculum.external_force_torque_levels = None
            self.curriculum.terrain_levels = None
            self.curriculum.stage_two = None



        self.events.reset_base.params['pose_range']['pitch']=(0,0)
        self.events.reset_base.params['pose_range']['roll']=(0,0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-.035, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.heading_command=False
        self.commands.base_velocity.rel_standing_envs = 0
