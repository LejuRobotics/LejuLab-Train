# velocity_amp_env_cfg
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from .mdp import mdp
from leju_robot.tasks.amp.envs import ManagerBasedRLAMPEnvCfg

##
# Pre-defined configs
##

from .terrains import ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/base",
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_L_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_R_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,track_pose=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##
# Single source for base_velocity lin_vel_x bounds; reward terms can derive scales from the same range.
_BASE_VEL_LIN_VEL_X = (-0.85, 0.85)
BASE_VEL_LIN_VEL_X_MAX = max(_BASE_VEL_LIN_VEL_X)

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = isaaclab_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.7,
        heading_command=True,
        heading_control_stiffness=0.3,
        debug_vis=True,
        ranges=isaaclab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=_BASE_VEL_LIN_VEL_X,
            lin_vel_y=(-0.45, 0.45),
            ang_vel_z=(-0.85, 0.85),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           joint_names=".*",
                                           preserve_order=True,
                                           scale=0.5,
                                           use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class AMPCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        root_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        root_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        rel_key_body_pos = ObsTerm(func=mdp.rel_key_body_pos_b)

        def __post_init__(self):
            # self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        joint_torques = ObsTerm(func=mdp.joint_torques)
        joint_accs = ObsTerm(func=mdp.joint_accs)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
            },
        )
        feet_contact_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["leg_[l,r]6_link"]
                )
            },
        )
        base_mass_rel = ObsTerm(
            func=mdp.rigid_body_masses,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        rigid_body_material = ObsTerm(
            func=mdp.rigid_body_material,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["leg_[l,r]6_link"])
            },
        )
        base_com = ObsTerm(
            func=mdp.base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        action_delay = ObsTerm(
            func=mdp.action_delay, params={"actuators_names": "motor"}
        )
        feet_heights = ObsTerm(
            func=mdp.feet_heights_bipeds,
            params={
                "sensor_cfg1": SceneEntityCfg("Feet_L_scanner"),
                "sensor_cfg2": SceneEntityCfg("Feet_R_scanner"),
            },
        )
        feet_air_times = ObsTerm(
            func=mdp.feet_air_time_obs,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names="leg_[l,r]6_link"
                ),
            },
        )

        ref_dof_pos_error = ObsTerm(
            func=mdp.ref_dof_pos_error,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AMPCfg = AMPCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_piecewise_exp, weight=6.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    stand_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd_last,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
        },
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-3)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_mean_acc_l2_mode, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.05)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"]), "threshold": 1.0},
    )

    #-- joint_deviation penalties
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint"
                ],
            )
        },
    )  

    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "leg_[l,r]6_joint"
                ],
            )
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "zarm_[l,r][2,]_joint"
                ],
            )
        },
    )

    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1_straight_only,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]2_joint"]),
            "angular_threshold": 0.1,
            "linear_threshold": 0.1,
        },
    )

    # -- feet
    feet_gait_swing = RewTerm(
        func=mdp.feet_gait_swing,
        weight=5.0,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity",
        },
    )

    feet_driven_arm_swing = RewTerm(
        func=mdp.arm_swing_gait_phase_reward_elbow,
        weight=1.0,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity", 
            "same_side_target": 1.0,
            "err_piecewise_split": 0.5,
        },
    )

    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward_floor_gate,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 10.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )

    feet_contact_force = RewTerm(
        func=mdp.contact_forces_penalty,
        weight=-0.003,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "threshold": 500, #350,
            "violation_max": 300,
            "violation_min": 0, #0
        },
    )
  
    feet_slide_vel = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )

    feet_slide_yaw = RewTerm(
        func=mdp.feet_slide_yaw,
        weight= -1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="leg_[l,r]6_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "command_name": "base_velocity",

        },
    )
    feet_both_grounded = RewTerm(
        func=mdp.both_feet_grounded,
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "velocity_threshold": 0.15,
            "max_grounded_time": 0.150,
        },
    )

    turn_dual_knee_flex = RewTerm(
        func=mdp.turn_swing_knee_flex_penalty,
        weight=-25.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "right_knee_cfg": SceneEntityCfg("robot", joint_names=["leg_r4_joint"]),
            "left_knee_cfg": SceneEntityCfg("robot", joint_names=["leg_l4_joint"]),
            "flex_knots": (0.2, 0.7, 1.0, 1.3),
        },
    )

    turn_in_place_leg1_abs = RewTerm(
        func=mdp.turn_in_place_leg1_abs_penalty,
        weight=-10.0,  
        params={"command_name": "base_velocity"},
    )

    feet_aligned_stance = RewTerm(
        func=mdp.feet_aligned_support_penalty_yaw,
        weight=-1000.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
            "min_fore_aft_separation_base": 0.01,
            "min_fore_aft_separation_per_vx": 0.3,
            "min_sep_max": 0.25,
            "command_ref": BASE_VEL_LIN_VEL_X_MAX,
        },
    )

    feet_y_distance_straight = RewTerm(func=mdp.feet_y_distance_straight, weight=-1.0, params={
        "command_name": "base_velocity"
    })
    fft_dof_symmetry = RewTerm(
        func=mdp.fft_dof_symmetry,
        weight=-0.015,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot"
            ),
            "joint_names_pairs":[
                    "leg_[l,r]6_joint",

                    "leg_[l,r]5_joint",

                    "leg_[l,r]4_joint",
                    "leg_[l,r]3_joint",
                    "zarm_[l,r]1_joint",
                    "zarm_[l,r]4_joint",
                    # "leg_[l,r]2_joint",
                    # "leg_[l,r]1_joint",
            ],
            "angular_threshold": 0.6,
            "command_name" : "base_velocity"
        },
        
    )
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.2, 1.4),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_.*_link", "zarm_.*_link"]
            ),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_base_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
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
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7),"yaw": (-3.14, 3.14),"pitch":(-0.1,0.1),"roll":(-0.1,0.1)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.01, 0.3),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-100.0, 100.0),
                "y": (-100.0, 100.0),
                "z": (- 100.0,  100.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-50.0, 50.0)},
            "probability": 0.01,  # Expect step = 1 / probability
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.radians(60.0),
        },
    )
    root_height_below_minimum = DoneTerm(
        mdp.root_height_below_minimum,
        params={"minimum_height": 0.35
        }
    )

##
# Environment configuration
##


@configclass
class LocomotionVelocityAMPRoughEnvCfg(ManagerBasedRLAMPEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # AMP observation space for Roban (21 joints):
        # dof_positions (21) + dof_velocities (21) + root_height (1) + 
        # gravity_vec (3) + root_lin_vel (3) + root_ang_vel (3) + 
        # 4 key_body_positions (4*3=12) = 64
        #todo check
        self.amp_observation_space = 64
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Fix PhysX collision stack size overflow
        self.sim.physx.gpu_collision_stack_size = 2**27
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
