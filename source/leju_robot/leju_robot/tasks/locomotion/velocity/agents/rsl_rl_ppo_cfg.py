from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = True
    enable_runner_safeguards: bool = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    runner_class_name = "RobotOnPolicyRunner"

@configclass
class RobanS14WalkPPORunnerCfg(RobotPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "robanS14_walk"
        self.max_iterations = 30000


@configclass
class KuavoS54WalkPPORunnerCfg(RobotPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS54_walk"
        self.max_iterations = 60000
        self.enable_runner_safeguards = True


@configclass
class KuavoS46WalkPPORunnerCfg(RobotPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS46_walk"
        self.max_iterations = 60000


@configclass
class KuavoS53WalkPPORunnerCfg(RobotPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_walk"
        self.max_iterations = 60000


@configclass
class RobanS17WalkPPORunnerCfg(RobotPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "robanS17_walk"
        self.max_iterations = 60000
        # Enable runner safeguards to prevent std<0 crashes in PPO sampling
        # (same protection as KuavoS54WalkPPORunnerCfg)
        self.enable_runner_safeguards = True
