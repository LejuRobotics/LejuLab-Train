import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from leju_robot.rsl_rl_extensions.utils.exporter import attach_onnx_metadata, export_policy_as_onnx

class RobotOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for motion tracking tasks with ONNX export support."""

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
