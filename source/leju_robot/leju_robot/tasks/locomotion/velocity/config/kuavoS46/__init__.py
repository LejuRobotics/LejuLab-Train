import gymnasium as gym

__all__ = ["KuavoS46_CFG"]

gym.register(
    id="Velocity-Rough-KuavoS46",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS46RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS46WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-KuavoS46-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS46RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:KuavoS46WalkPPORunnerCfg",
    },
)
