from isaaclab.utils import configclass

from leju_robot.tasks.locomotion.velocity.config.kuavoS54.rough_env_cfg import (
    KuavoS54RoughEnvCfg,
)
from leju_robot.tasks.locomotion.velocity.config.kuavoS53.kuavoS53 import KuavoS53_CFG


@configclass
class KuavoS53RoughEnvCfg(KuavoS54RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Replace robot asset
        self.scene.robot = KuavoS53_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Replace joint order (same 27 DOF structure, points to S53 preserve_joint_order)
        self.actions.joint_pos.joint_names = KuavoS53_CFG.preserve_joint_order.joint_names
        self.observations.policy.joint_pos_rel.params = {"asset_cfg": KuavoS53_CFG.preserve_joint_order}
        self.observations.policy.joint_vel_rel.params = {"asset_cfg": KuavoS53_CFG.preserve_joint_order}
        self.observations.critic.joint_pos_rel.params = {"asset_cfg": KuavoS53_CFG.preserve_joint_order}
        self.observations.critic.joint_vel_rel.params = {"asset_cfg": KuavoS53_CFG.preserve_joint_order}
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS53_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        }


@configclass
class KuavoS53RoughEnvCfg_PLAY(KuavoS53RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 1e9

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.events.physics_material = None
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS53_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None

        self.events.reset_robot_joints = None

        self.events.push_robot = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
        self.commands.base_velocity.ranges.heading = (-0, 0)
        self.commands.base_velocity.heading_command = False
