import gymnasium as gym

__all__ = ["KuavoS53_CFG"]

gym.register(
    id="Velocity-Rough-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS53RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS53WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS53RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS53WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS53FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS53WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS53FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS53WalkPPORunnerCfg",
    },
)
