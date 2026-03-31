import gymnasium as gym

# from .robanS14 import RobanS14_CFG, PRESERVE_JOINT_ORDER_ASSET_CFG

__all__ = ["RobanS14_CFG", "PRESERVE_JOINT_ORDER_ASSET_CFG"]

gym.register(
    id="Velocity-Rough-RobanS14",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobanS14RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS14WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-RobanS14-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobanS14RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS14WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-RobanS14",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobanS14FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS14WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-RobanS14-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobanS14FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS14WalkPPORunnerCfg",
    },
)
