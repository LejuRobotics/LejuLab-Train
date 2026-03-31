from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from leju_robot.tasks.tracking.config.robanS14.new_year_dance.robanS14 import RobanS14_ACTION_SCALE, RobanS14_CYLINDER_CFG
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from leju_robot.assets.motion_data import MOTION_DIR
import leju_robot.tasks.tracking.mdp as mdp
##
# Scene definition
##

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = RobanS14_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=40.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        motion_file=f"{MOTION_DIR}/mimic/npz_data/robanS14_new_year_dance_50fps.npz",
        asset_name="robot",
        anchor_body = "base_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
        body_names = [
            "base_link",
            "waist_yaw_link",
            "leg_l2_link",
            "leg_l4_link",
            "leg_l6_link",
            "leg_r2_link",
            "leg_r4_link",
            "leg_r6_link",
            "zarm_l2_link",
            "zarm_l4_link",
            "zarm_r2_link",
            "zarm_r4_link",
        ],
    )

@configclass
class CommandsCfg_PLAY(CommandsCfg):
    """Command specifications for PLAY mode - simplified logic."""

    def __post_init__(self):
        super().__post_init__()
        self.motion = mdp.MotionCommandPlayCfg(
            motion_file=self.motion.motion_file,
            asset_name=self.motion.asset_name,
            anchor_body=self.motion.anchor_body,
            resampling_time_range=self.motion.resampling_time_range,
            debug_vis=self.motion.debug_vis,
            pose_range=self.motion.pose_range,
            velocity_range=self.motion.velocity_range,
            joint_position_range=self.motion.joint_position_range,
            body_names=self.motion.body_names,
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionResidualsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=0.5,
        use_default_offset=True
    )
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[".*"],
    #     preserve_order=True,
    #     use_default_offset=True
    # )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_target_pos = ObsTerm(
            func=mdp.motion_target_pos, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_target_pos = ObsTerm(func=mdp.motion_target_pos, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        projected_gravity = ObsTerm( func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 5

    # @configclass
    # class VisCfg(ObsGroup):
        # joint_torques_zarm1 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_[lr]1_joint"])})
        # joint_torques_zarm4 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_[lr]4_joint"])})
        # joint_torques_leg5 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr]5_joint"])})
        # joint_torques_lleg6 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_l6_joint"])})
        # joint_torques_rleg6 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_r6_joint"])})
        # joint_torques_leg1 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr]1_joint"])})
        # joint_torques_leg4 = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr]4_joint"])})
        # joint_torques = ObsTerm(func=mdp.joint_torque, params={"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG})
        # feet_contact_forces = ObsTerm(
        #     func=mdp.feet_contact_forces,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces",
        #             body_names=["leg_l6_link", "leg_r6_link"]
        #         )
        #     }
        # )
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    # visit: VisCfg = VisCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.3),
            "dynamic_friction_range": (0.3, 1.3),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.035, 0.035), "y": (-0.06, 0.06), "z": (-0.06, 0.06)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-4]_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-4]_link"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # init_push_manager = EventTerm(
    #     func=mdp.init_push_manager,
    #     mode="startup",
    #     params={
    #         "disturbance_categories": [
    #             {
    #                 # Short / small - Hips, Feet
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "leg_l6_link", "leg_r6_link"]),
    #                 "force_range_xy": (0.0, 5.0),
    #                 "force_range_z": (0.0, 5.0),
    #                 "torque_range_xy": (0.0, 0.25),
    #                 "torque_range_z": (0.0, 0.25),
    #                 "duration_range_on": (0.25, 2.0),
    #                 "interval_range_off": (1.0, 3.0),
    #             },
    #             {
    #                 # Long / small - hand
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["zarm_l1_link", "zarm_r1_link", "zarm_l4_link", "zarm_r4_link"]),
    #                 "force_range_xy": (0.0, 5.0),
    #                 "force_range_z": (0.0, 5.0),
    #                 "torque_range_xy": (0.0, 0.25),
    #                 "torque_range_z": (0.0, 0.25),
    #                 "duration_range_on": (2.0, 10.0),
    #                 "interval_range_off": (1.0, 3.0),
    #             },
    #             {
    #                 # Short / large - Pelvis
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #                 "force_range_xy": (90.0, 150.0),
    #                 "force_range_z": (0.0, 10.0),
    #                 "torque_range_xy": (0.0, 15.0),
    #                 "torque_range_z": (0.0, 15.0),
    #                 "duration_range_on": (0.1, 0.1),
    #                 "interval_range_off": (12.0, 15.0),
    #             },
    #         ],
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78)}},
    )
    
    # apply_disturbance_unified = EventTerm(
    #     func=mdp.push_by_impulse_from_force_duration,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params={
    #         "disturbance_categories": [
    #             {
    #                 # Short / small - Hips, Feet
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "leg_l6_link", "leg_r6_link"]),
    #                 "force_range_xy": (0.0, 5.0),
    #                 "force_range_z": (0.0, 5.0),
    #                 "torque_range_xy": (0.0, 0.25),
    #                 "torque_range_z": (0.0, 0.25),
    #                 "duration_range_on": (0.25, 2.0),
    #                 "interval_range_off": (1.0, 3.0),
    #             },
    #             {
    #                 # Long / small - hand
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["zarm_l1_link", "zarm_r1_link", "zarm_l4_link", "zarm_r4_link"]),
    #                 "force_range_xy": (0.0, 5.0),
    #                 "force_range_z": (0.0, 5.0),
    #                 "torque_range_xy": (0.0, 0.25),
    #                 "torque_range_z": (0.0, 0.25),
    #                 "duration_range_on": (2.0, 10.0),
    #                 "interval_range_off": (1.0, 3.0),
    #             },
    #             {
    #                 # Short / large - Pelvis
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #                 "force_range_xy": (90.0, 150.0),
    #                 "force_range_z": (0.0, 10.0),
    #                 "torque_range_xy": (0.0, 15.0),
    #                 "torque_range_z": (0.0, 15.0),
    #                 "duration_range_on": (0.1, 0.1),
    #                 "interval_range_off": (12.0, 15.0),
    #             },
    #         ],
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-10.0,
        params={"soft_ratio": 1, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!leg_l4_link$)(?!leg_r4_link$)(?!leg_l6_link$)(?!leg_r6_link$)(?!zarm_l4_link$)(?!zarm_r4_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )

    # motion_feet_pos = RewTerm(
    #     func=mdp.motion_feet_position_error_exp,
    #     weight=2.0,
    #     params={"command_name": "motion", "std": 0.04, "body_names": ["leg_l6_link","leg_r6_link"]},
    # )


    motion_hand_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.04, "body_names": ["zarm_l4_link","zarm_r4_link"]},
    )

    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.04, "body_names": ["leg_l4_link","leg_r4_link"]},
    )
    motion_knee_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 1.0, "body_names": ["leg_l4_link","leg_r4_link"]},
    )

    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_l6_link", "leg_r6_link"
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["leg_l6_link", "leg_r6_link"],
            ),
            "force_velocity_threshold": 50,
        },
    )


    feet_slide_vel = RewTerm(
        func=mdp.body_slide_vel,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_[l,r]6_link"]),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_[l,r]6_link"]),
            "contact_threshold": 200.0
        },
    )

    com_balance_reward = RewTerm(
        func=mdp.com_balance_when_stand,
        weight=0.7,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
            ),
            "std": 0.05,  # 标准差（米），质心偏移超过此值奖励快速衰减
        },
    )
    # penalize_feet_height = RewTerm(
    #     func=mdp.penalize_feet_height,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["leg_l6_link", "leg_r6_link"]),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["leg_[l,r]6_link"]),
    #         "swing_force_ratio": 0.15,
    #         "support_force_ratio": 0.5,
    #         "single_support_target": 0.11,
    #         "single_support_scale": 5.0,
    #     },
    # )

    # twisted_feet = RewTerm(
    #     func=mdp.twisted_feet,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["leg_l6_link", "leg_r6_link"]),
    #         "shank_cfg": SceneEntityCfg(
    #             "robot", body_names=["leg_[l,r]4_link"]),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["leg_[l,r]6_link"]),
    #         "contact_threshold": 30.0,
    #         "horizontal_threshold": 0.1,
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "leg_l6_link",
                "leg_r6_link",
                "zarm_l4_link",
                "zarm_r4_link",
            ],
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class RobanS14FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        self.actions.joint_pos.scale = RobanS14_ACTION_SCALE
        self.actions.joint_pos.joint_names = RobanS14_CYLINDER_CFG.preserve_joint_order.joint_names
        self.observations.policy.joint_pos.params = {"asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order}
        self.observations.policy.joint_vel.params = {"asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order}
        self.observations.critic.joint_pos.params = {"asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order}
        self.observations.critic.joint_vel.params = {"asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order}

@configclass
class RobanS14FlatEnvCfg_PLAY(RobanS14FlatEnvCfg):
    """Configuration for PLAY mode - simplified command logic."""
    
    commands: CommandsCfg_PLAY = CommandsCfg_PLAY()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 1e9
        
        # disable randomization for play (make environment deterministic)
        # disable observation noise
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        
        self.events.physics_material = None
        self.events.add_joint_default_pos.params ={
            "asset_cfg": RobanS14_CYLINDER_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.base_com = None
        self.events.add_base_mass = None
        self.events.link_com = None
        self.events.add_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        
        # remove random pushing events
        self.events.push_robot = None
        # self.events.base_external_force_torque = None
        
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

