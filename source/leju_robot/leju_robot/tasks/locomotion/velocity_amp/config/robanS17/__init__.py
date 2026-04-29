import gymnasium as gym

from . import agents

gym.register(
    id="Velocity-AMP-RobanS17",
    entry_point="leju_robot.tasks.locomotion.velocity_amp.velocity_amp_env:LocomotionVelocityAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.roban_env_cfg:RobanS2MixEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_flat_amp_mse_cfg.yaml",
    }
)

gym.register(
    id="Velocity-AMP-RobanS17-Play",
    entry_point="leju_robot.tasks.locomotion.velocity_amp.velocity_amp_env:LocomotionVelocityAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.roban_env_cfg:RobanS2MixEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_flat_amp_mse_cfg_play.yaml",
    }
)