import gymnasium as gym

gym.register(
    id="Tracking-Standup-Flat-RobanS14",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobanS14FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:RobanS14StandupPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Standup-Flat-RobanS14-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobanS14FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:RobanS14StandupPPORunnerCfg",
    },
)