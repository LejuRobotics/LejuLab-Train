import gymnasium as gym

__all__ = ["KuavoS54_CFG", "PRESERVE_JOINT_ORDER_ASSET_CFG"]

gym.register(
    id="Velocity-Rough-KuavoS54",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS54RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS54WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-KuavoS54-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS54RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS54WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-KuavoS54",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS54FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS54WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-KuavoS54-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS54FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS54WalkPPORunnerCfg",
    },
)
