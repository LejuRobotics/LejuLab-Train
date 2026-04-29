from collections.abc import Sequence

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg


@configclass
class ManagerBasedRLAMPEnvCfg(ManagerBasedRLEnvCfg):

    motion_file: str = MISSING

    key_body_names: Sequence[str] = MISSING

    reference_body: str = MISSING

    num_amp_observations = 2
    # AMP observation space for Roban (21 joints): 21+21+1+3+3+3+12=64
    amp_observation_space = 64  
