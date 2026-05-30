import gymnasium as gym

gym.register(
    id="Velocity-Rough-RobanS17",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobanS17RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS17WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-RobanS17-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobanS17RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS17WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-RobanS17",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobanS17FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS17WalkPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-RobanS17-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RobanS17FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.locomotion.velocity.agents.rsl_rl_ppo_cfg:RobanS17WalkPPORunnerCfg",
    },
)
