from isaaclab.utils import configclass

from .rough_env_cfg import KuavoS53RoughEnvCfg
from leju_robot.tasks.locomotion.velocity.config.kuavoS53.kuavoS53 import KuavoS53_CFG


@configclass
class KuavoS53FlatEnvCfg(KuavoS53RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.Feet_L_scanner = None
        self.scene.Feet_R_scanner = None
        self.observations.critic.height_scan = None
        self.observations.critic.feet_heights = None
        self.curriculum.terrain_levels = None


class KuavoS53FlatEnvCfg_PLAY(KuavoS53FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1e9

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.events.randomize_rigid_body_com = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_link_mass = None
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS53_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.scale_joint_parameters = None

        self.events.reset_robot_joints = None

        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-0, 0)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = True
