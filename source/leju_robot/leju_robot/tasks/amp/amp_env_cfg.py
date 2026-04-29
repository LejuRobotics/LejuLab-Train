from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp
from .envs import ManagerBasedRLAMPEnvCfg


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.MyJointPositionToLimitsActionCfg(asset_name="robot",
                                           joint_names=".*",
                                           preserve_order=True,
                                           scale=0.5,
                                           use_default_offset=True)
    # joint_pos = mdp.JointPositionToLimitsActionCfg(asset_name="robot",
    #                                                joint_names=PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names,)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        root_orient = ObsTerm(mdp.tangent_and_normal)
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        rel_key_body_pos = ObsTerm(func=mdp.rel_key_body_pos)

        def __post_init__(self):
            # self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class AMPCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            # self.history_length = 2


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    amp: AMPCfg = AMPCfg()


@configclass
class RewardsCfg:
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class EventCfg:

    #startup

    # reset
    reset_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    reset_random = EventTerm(
        func=mdp.reset_scene_strategy_random,
        params={
            "start": False,
        },
        mode="reset"
    )

    # interval


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(
        mdp.terminations.root_height_below_minimum,
        params={"minimum_height": 0.5}
    )


##
# Environment configuration
##


@configclass
class AMPEnvCfg(ManagerBasedRLAMPEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Fix PhysX collision stack size overflow
        self.sim.physx.gpu_collision_stack_size = 2**27

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        
        # self.observations.amp.history_length = self.num_amp_observations



