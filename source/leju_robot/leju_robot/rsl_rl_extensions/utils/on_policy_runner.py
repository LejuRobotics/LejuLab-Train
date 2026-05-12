import os

import torch

from rsl_rl.env import VecEnv
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from leju_robot.rsl_rl_extensions.utils.exporter import attach_onnx_metadata, export_policy_as_onnx


class _ClippedEmpiricalNormalization(EmpiricalNormalization):
    """EmpiricalNormalization with output clipping to prevent extreme values."""

    def forward(self, x):
        # Inline parent forward to avoid super() / class method calls that
        # TorchScript cannot resolve after __class__ swap.
        if self.training:
            self.update(x)
        return torch.clamp((x - self._mean) / (self._std + self.eps), min=-5.0, max=5.0)


class RobotOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with ONNX export and NaN reward protection."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)

        # Safeguards (NaN reward handling, reward clamp, normalizer output clamp)
        # are opt-in via cfg flag. Only Velocity-KuavoS54 enables them; tracking
        # and other tasks early-return here so behavior matches master exactly.
        if not train_cfg.get("enable_runner_safeguards", False):
            return

        # Wrap env.step to replace NaN rewards with 0, preventing NaN from
        # corrupting PPO's GAE computation. NaN rewards arise because Isaac Lab
        # computes rewards before checking termination conditions — when
        # invalid_state terminates an environment, its reward for that step
        # is already NaN.
        _original_step = self.env.step

        def _safe_step(actions):
            obs, rewards, dones, infos = _original_step(actions)
            rewards = torch.nan_to_num(rewards, nan=0.0)
            rewards = torch.clamp(rewards, min=-1000.0, max=1000.0)
            return obs, rewards, dones, infos

        self.env.step = _safe_step

        # Swap normalizer class to clip extreme values after normalization.
        # EmpiricalNormalization has no output clipping — when _std is very small
        # (e.g., push_force is zero most of the time), normalized values can be
        # extreme, causing the critic to output inf → inf value loss.
        # Using __class__ swap instead of monkey-patching forward, because the
        # ONNX exporter deepcopies the normalizer — closures don't survive that.
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.__class__ = _ClippedEmpiricalNormalization
        if isinstance(self.privileged_obs_normalizer, EmpiricalNormalization):
            self.privileged_obs_normalizer.__class__ = _ClippedEmpiricalNormalization

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        export_policy_as_onnx(
            self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
        )
        try:
            attach_onnx_metadata(self.env.unwrapped, "none", path=policy_path, filename=filename)
        except Exception:
            pass
