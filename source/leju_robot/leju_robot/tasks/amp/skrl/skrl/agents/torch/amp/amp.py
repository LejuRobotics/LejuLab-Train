from typing import Any, Callable, Mapping, Optional, Tuple, Union

import copy
import itertools
import math
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# fmt: off
# [start-config-dict-torch]
AMP_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {"max_lr": 0.003},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3}),setting max_lr very importance, a large learning rate would cause gradient exploding problem

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "amp_state_preprocessor": None,         # AMP state preprocessor class (see skrl.resources.preprocessors)
    "amp_state_preprocessor_kwargs": {},    # AMP state preprocessor's kwargs (e.g. {"size": env.amp_observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,          # entropy loss scaling factor
    "value_loss_scale": 2.5,            # value loss scaling factor
    "discriminator_loss_scale": 5.0,    # discriminator loss scaling factor

    "amp_batch_size": 512,                  # batch size for updating the reference motion dataset
    "task_reward_weight": 0.0,              # task-reward weight (wG)
    "style_reward_weight": 1.0,             # style-reward weight (wS) - fallback if mode-specific not set
    "style_reward_weight_mode0": None,      # style-reward weight for command_state==0 (walking), None=use style_reward_weight
    "style_reward_weight_mode2": None,      # style-reward weight for command_state==2 (walking+arm), None=use style_reward_weight
    "discriminator_batch_size": 0,          # batch size for computing the discriminator loss (all samples if 0)
    "discriminator_reward_scale": 2,                    # discriminator reward scaling factor
    "discriminator_logit_regularization_scale": 0.05,   # logit regularization scale factor for the discriminator loss
    "discriminator_gradient_penalty_scale": 5,          # gradient penalty scaling factor for the discriminator loss
    "discriminator_weight_decay_scale": 0.0001,         # weight decay scaling factor for the discriminator loss

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance


    "discriminator_loss_type": 'MSE',   # 'Wasserstein',"BCE","MSE"
    "soft_boundary_constraint_scale": 0. , #a hyperparameter that controls the range of ouput boundaries,(0.1, 0.5) is a proper range for selection
    "with_reply_sample": True,      # use reply policy data to train discriminator

    "use_symmetry_loss":False,
    "symmetry_scale":1,
    "max_style_reward_scale":0, # will clip style reward ,max value is n times of style_reward mean value 
    "style_reward_scale_mode2": 1.0,  # scale style reward for command_state==2


    "disc_sym_loss_mode":0, # 0 would not use discriminator symmetry loss ,1 means use policy discriminator sym loss,2 means use expert discriminator sym loss,3 means use expert and policy discriminator sym loss 

    # Joint DOF order conversion for "upper-body masking" on AMP observations (command_state==2).
    # If gym order != lab order, set these to temporarily reorder dof_pos/dof_vel before/after masking.
    # If gym order == lab order (IsaacLab default), keep as identity: list(range(n_dof)).
    # - gym2lab: gym_index -> lab_index
    # - lab2gym: lab_index -> gym_index
    #
    # RobanS2 (21 joints, lab order = PRESERVE_JOINT_ORDER_ASSET_CFG):
# 需确认是否需要
    "amp_dof_gym2lab": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "amp_dof_lab2gym": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

    }
}
# [end-config-dict-torch]
# fmt: on




class AMP(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        amp_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        motion_dataset: Optional[Memory] = None,
        reply_buffer: Optional[Memory] = None,
        collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
        collect_observation: Optional[Callable[[], torch.Tensor]] = None,
        discriminator_obs_history_length :int = 2,
    ) -> None:
        """Adversarial Motion Priors (AMP)

        https://arxiv.org/abs/2104.02180

        The implementation is adapted from the NVIDIA IsaacGymEnvs
        (https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/learning/amp_continuous.py)

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param amp_observation_space: AMP observation/state space or shape (default: ``None``)
        :type amp_observation_space: int, tuple or list of int, gymnasium.Space or None
        :param motion_dataset: Reference motion dataset: M (default: ``None``)
        :type motion_dataset: skrl.memory.torch.Memory or None
        :param reply_buffer: Reply buffer for preventing discriminator overfitting: B (default: ``None``)
        :type reply_buffer: skrl.memory.torch.Memory or None
        :param collect_reference_motions: Callable to collect reference motions (default: ``None``)
        :type collect_reference_motions: Callable[[int], torch.Tensor] or None
        :param collect_observation: Callable to collect observation (default: ``None``)
        :type collect_observation: Callable[[], torch.Tensor] or None

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(AMP_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        self._discriminator_history_length = discriminator_obs_history_length

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.discriminator = self.models.get("discriminator", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()
            if self.discriminator is not None:
                self.discriminator.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._discriminator_loss_scale = self.cfg["discriminator_loss_scale"]

        self._learning_rate = self.cfg["learning_rate"]
        self._discriminator_learning_rate = self.cfg.get("discriminator_learning_rate", self._learning_rate)  # 新增：获取判别器学习率
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]
        self._amp_state_preprocessor = self.cfg["amp_state_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._amp_batch_size = self.cfg["amp_batch_size"]
        self._task_reward_weight = self.cfg["task_reward_weight"]
        self._style_reward_weight = self.cfg["style_reward_weight"]
        # 分模式 style_reward_weight，如果未设置则 fallback 到 style_reward_weight
        self._style_reward_weight_mode0 = self.cfg.get("style_reward_weight_mode0")
        if self._style_reward_weight_mode0 is None:
            self._style_reward_weight_mode0 = self._style_reward_weight
        self._style_reward_weight_mode2 = self.cfg.get("style_reward_weight_mode2")
        if self._style_reward_weight_mode2 is None:
            self._style_reward_weight_mode2 = self._style_reward_weight

        self._discriminator_batch_size = self.cfg["discriminator_batch_size"]
        self._discriminator_reward_scale = self.cfg["discriminator_reward_scale"]
        self._discriminator_logit_regularization_scale = self.cfg["discriminator_logit_regularization_scale"]
        self._discriminator_gradient_penalty_scale = self.cfg["discriminator_gradient_penalty_scale"]
        self._discriminator_weight_decay_scale = self.cfg["discriminator_weight_decay_scale"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        self._soft_boundary_constraint_scale =self.cfg["soft_boundary_constraint_scale"]
        self._with_reply_sample = self.cfg["with_reply_sample"]
        self._max_style_reward_scale = self.cfg['max_style_reward_scale']
        self._style_reward_scale_mode2 = self.cfg.get("style_reward_scale_mode2", 1.0)

        self._discriminator_loss_type = self.cfg['discriminator_loss_type']

        if self._discriminator_loss_type not in ["Wasserstein", "BCE", "MSE"]:
            raise ValueError('the discriminator loss type must be the one of ["Wasserstein", "BCE", "MSE"]')


        self._use_symmetry_loss = self.cfg['use_symmetry_loss']
        self._symmetry_scale = self.cfg["symmetry_scale"]

        self._disc_sym_loss_mode = self.cfg["disc_sym_loss_mode"]

        print("[info] amp agent cfg: \n", self.cfg)

        # AMP requires a reference motion sampler when using a motion dataset
        if self.motion_dataset is not None and self.collect_reference_motions is None:
            raise ValueError("collect_reference_motions must be provided when motion_dataset is not None")
        if self.motion_dataset is not None and self._with_reply_sample and self.reply_buffer is None:
            raise ValueError("reply_buffer must be provided when with_reply_sample is True and motion_dataset is not None")

        # AMP dof order mapping tensors 
        # NOTE: These are only used to apply upper-body masking using gym indices, while the env may output AMP dofs in lab order.
        self._amp_dof_gym2lab = torch.tensor(self.cfg.get("amp_dof_gym2lab", []), dtype=torch.long, device=self.device)
        self._amp_dof_lab2gym = torch.tensor(self.cfg.get("amp_dof_lab2gym", []), dtype=torch.long, device=self.device)

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None and self.discriminator is not None:
            # 修改：为策略和价值网络创建共享优化器
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters()),
                lr=self._learning_rate,
            )

            # 新增：为判别器创建独立优化器
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self._discriminator_learning_rate,
            )

            # 修改：为策略/价值网络创建学习率调度器
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

                # 新增：为判别器创建独立的学习率调度器
                self.discriminator_scheduler = self._learning_rate_scheduler(
                    self.discriminator_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            # 修改：更新checkpoint模块
            self.checkpoint_modules["optimizer"] = self.optimizer
            self.checkpoint_modules["discriminator_optimizer"] = self.discriminator_optimizer  # 新增

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        if self._observation_preprocessor:
            self._observation_preprocessor = self._observation_preprocessor(**self.cfg["observation_preprocessor_kwargs"])
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self.cfg["amp_state_preprocessor_kwargs"])
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self.memory.create_tensor(name="amp_states", size=self.amp_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)

        self.tensors_names = [
            "states",
            "observations",
            "actions",
            "rewards",
            "next_states",
            "next_observations",
            "terminated",
            "log_prob",
            "values",
            "returns",
            "advantages",
            "amp_states",
            "next_values",
        ]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            if self._with_reply_sample:
                self.reply_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)

            # initialize motion dataset
            for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_states = None
        self._current_observations = None



        #create temporary variables needed for storage symmetry data
        self._current_log_prob_sym = None
        self._current_action_sym = None
        self._cuurent_state_sym = None
        self._current_action_mean_sym = None
        self._current_observations_sym = None



    def act(self, observations: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # use collected states

        self._current_observations_sym = self.flip_actor_obs(observations)

        if self._current_observations is not None:
            observations = self._current_observations
        observations = self._observation_preprocessor(observations)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"observations": observations}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"observations": observations}, role="policy")
        self._current_log_prob = log_prob

        if self._use_symmetry_loss:
            actions_sym = self.flip_action(actions)
            self._current_log_prob_sym = log_prob.detach().clone()
            self._current_action_sym = actions_sym
            self._current_action_mean_sym = self.flip_action(outputs["mean_actions"])


        return actions, log_prob, outputs

    def record_transition(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        super().record_transition(
            observations, states, actions, rewards, next_observations, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            amp_states = infos["amp_obs"]

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({
                    "observations": self._observation_preprocessor(observations),
                    "states": self._state_preprocessor(states)
                }, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # compute next values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                next_values, _, _ = self.value.act({
                    "observations": self._observation_preprocessor(next_observations),
                    "states": self._state_preprocessor(next_states)
                }, role="value")
                next_values = self._value_preprocessor(next_values, inverse=True)
                if "terminate" in infos:
                    next_values *= infos["terminate"].view(-1, 1).logical_not()  # compatibility with IsaacGymEnvs
                else:
                    next_values *= terminated.view(-1, 1).logical_not()
                
                

            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                amp_states=amp_states,
                next_values=next_values,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    amp_states=amp_states,
                    next_values=next_values,
                )
            if self._use_symmetry_loss:

                self._cuurent_state_sym=self.flip_critic_obs(states)
                next_observations_sym = self.flip_actor_obs(next_observations)
                next_states_sym = self.flip_critic_obs(next_states)
                self._amp_state_sym = self.flip_amp_obs(amp_states)

                # compute values
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    values_sym, _, _ = self.value.act({
                        "observations": self._observation_preprocessor(self._current_observations_sym),
                        "states": self._state_preprocessor(self._cuurent_state_sym)
                    }, role="value")
                    values_sym = self._value_preprocessor(values_sym, inverse=True)

                # time-limit (truncation) bootstrapping
                if self._time_limit_bootstrap:
                    rewards_sym = rewards - self._discount_factor * values * truncated
                    rewards_sym += self._discount_factor * values_sym * truncated

                # compute next values
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    next_values_sym, _, _ = self.value.act({
                        "observations": self._observation_preprocessor(next_observations_sym),
                        "states": self._state_preprocessor(next_states_sym)
                    }, role="value")
                    next_values_sym = self._value_preprocessor(next_values_sym, inverse=True)
                    if "terminate" in infos:
                        next_values_sym *= infos["terminate"].view(-1, 1).logical_not()  # compatibility with IsaacGymEnvs
                    else:
                        next_values_sym *= terminated.view(-1, 1).logical_not()
                

                self.memory.add_samples(
                    observations=self._current_observations_sym,
                    states=self._cuurent_state_sym,
                    actions=self._current_action_sym,
                    rewards=rewards_sym,
                    next_observations=next_observations_sym,
                    next_states=next_states_sym,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob_sym,
                    values=next_values_sym,
                    amp_states=self._amp_state_sym,
                    next_values=next_values_sym,
                )
                for memory in self.secondary_memories:
                    self.memory.add_samples(
                    observations=self._current_observations_sym,
                    states=self._cuurent_state_sym,
                    actions=self._current_action_sym,
                    rewards=rewards_sym,
                    next_observations=next_observations_sym,
                    next_states=next_states_sym,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob_sym,
                    values=next_values_sym,
                    amp_states=self._amp_state_sym,
                    next_values=next_values_sym,
                    )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.collect_observation is not None:
            self._current_observations = self.collect_observation()

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages
    
        def compute_gae_group(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
            group_idx:torch.Tensor = None,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE) with grouping

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            for idx in torch.unique(group_idx):
                mask = group_idx == idx
                advantages_masked = advantages[mask]
                advantages[mask] = (advantages_masked - advantages_masked.mean()) / (advantages_masked.std() + 1e-8)
            return returns, advantages

        # update dataset of reference motions
        if self.motion_dataset is None or self.collect_reference_motions is None:
            raise ValueError("motion_dataset and collect_reference_motions must be provided for AMP updates")
        self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        # compute combined rewards
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")


        # 当站立或下蹲弯腰时,去掉discriminator的奖励与LOSS
        observations = self.memory.get_tensor_by_name("observations")

        # 兼容不包含 command_state 的观测：Roban 新观测(72/51)缺失时按 mode0 处理
        obs_dim = observations.shape[-1]
        has_command_state = obs_dim not in (72, 51)
        if has_command_state:
            command_state = observations[:, :, 9]
        else:
            command_state = torch.zeros_like(observations[:, :, 0])
        lower_body_only_mask = command_state == 2
        amp_states_for_reward = self.scale_amp_obs(self._amp_state_preprocessor(amp_states))
        # if torch.any(lower_body_only_mask):
        #     amp_states_for_reward = amp_states_for_reward.clone()
        #     # Convert (lab -> gym) before applying masking indices (13:27, 40:54, 64:70) which are defined in gym order
        #     amp_states_for_reward_gym = self._amp_states_lab_to_gym(amp_states_for_reward)
        #
        #     amp_shape = amp_states_for_reward_gym.shape
        #     amp_states_flat = amp_states_for_reward_gym.view(-1, amp_shape[-1])
        #     lower_body_only_mask_flat = lower_body_only_mask.reshape(-1)
        #     single_amp_obs = self.amp_observation_space.shape[0] // self._discriminator_history_length
        #     dof_pos_sl, dof_vel_sl, key_body_sl = self._get_upper_body_mask_slices(single_amp_obs)
        #     for i in range(self._discriminator_history_length):
        #         start_idx = i * single_amp_obs
        #         # mask掉上半身关节位置
        #         amp_states_flat[lower_body_only_mask_flat, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
        #         # mask掉上半身关节速度
        #         amp_states_flat[lower_body_only_mask_flat, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
        #         # mask掉上半身key body位置（手部）
        #         amp_states_flat[lower_body_only_mask_flat, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
        #     amp_states_for_reward_gym = amp_states_flat.view(amp_shape)
        #     # Convert back (gym -> lab) to keep discriminator inputs consistent with env AMP observation order
        #     amp_states_for_reward = self._amp_states_gym_to_lab(amp_states_for_reward_gym)

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            amp_logits, _, _ = self.discriminator.act(
                {"states": amp_states_for_reward}, role="discriminator"
            )
            if torch.any(torch.isnan(amp_logits)) or torch.any(torch.isinf(amp_logits)):
                logger.warning("amp_logits包含NaN或Inf，使用nan_to_num修复")
                amp_logits = torch.nan_to_num(amp_logits, nan=0.0, posinf=50.0, neginf=-50.0)

            match self._discriminator_loss_type:
                case "Wasserstein":
                    # 防止exp溢出：clip amp_logits到合理范围（exp(88)约为inf）
                    amp_logits_clipped = torch.clamp(amp_logits, min=-50.0, max=50.0)
                    style_reward = torch.exp(amp_logits_clipped)
                    # 检查NaN和inf
                    if torch.any(torch.isnan(style_reward)) or torch.any(torch.isinf(style_reward)):
                        logger.warning("style_reward包含NaN或Inf，使用nan_to_num修复")
                        style_reward = torch.nan_to_num(style_reward, nan=0.0, posinf=1e6, neginf=0.0)
                    if self._max_style_reward_scale > 0:
                        style_reward_mean = style_reward.mean()
                        if torch.isnan(style_reward_mean) or torch.isinf(style_reward_mean):
                            style_reward_mean = 1.0
                        style_reward = torch.clip(style_reward, max=style_reward_mean*self._max_style_reward_scale)
                case "BCE":
                    style_reward = -torch.log(
                        torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
                    )
                case "MSE":
                    style_reward = torch.clamp(
                        1 - (1 / 4) * torch.square(amp_logits - 1), min=0
                    )
            style_reward *= self._discriminator_reward_scale
            style_reward = style_reward.view(rewards.shape)

        if self._disc_sym_loss_mode == 0 :
            style_reward[1::2]=style_reward[0::2]

        #当站立或下蹲弯腰时,去掉discriminator的奖励与LOSS
        # 创建行走任务的mask (只有行走时才使用AMP风格约束)  9=command_state index on policy observation
        observations_command_state_ = command_state.unsqueeze(-1)
        walking_mask = ((observations_command_state_ == 0) | (observations_command_state_ == 2)).float().detach()  # detach避免mask产生梯度
        style_reward = style_reward * walking_mask
        if self._style_reward_scale_mode2 != 1.0:
            style_reward = torch.where(
                command_state.unsqueeze(-1) == 2,
                style_reward * self._style_reward_scale_mode2,
                style_reward,
            )

        # 记录分模式的 style reward / logits 统计信息
        def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            values_flat = values.reshape(-1)
            mask_flat = mask.reshape(-1)
            if torch.any(mask_flat):
                selected = values_flat[mask_flat]
                selected = selected[torch.isfinite(selected)]
                if selected.numel() > 0:
                    return selected.mean().item()
            return 0.0

        def _safe_masked_std(values: torch.Tensor, mask: torch.Tensor) -> float:
            values_flat = values.reshape(-1)
            mask_flat = mask.reshape(-1)
            if torch.any(mask_flat):
                selected = values_flat[mask_flat]
                selected = selected[torch.isfinite(selected)]
                if selected.numel() > 0:
                    return selected.std().item()
            return 0.0

        def _safe_masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
            values_flat = values.reshape(-1)
            mask_flat = mask.reshape(-1)
            if torch.any(mask_flat):
                selected = values_flat[mask_flat]
                selected = selected[torch.isfinite(selected)]
                if selected.numel() > 0:
                    return torch.quantile(selected, q).item()
            return 0.0

        command_state_mask0 = command_state == 0
        command_state_mask1 = command_state == 1
        command_state_mask2 = command_state == 2
        has_mode0 = bool(torch.any(command_state_mask0).item())
        has_mode1 = bool(torch.any(command_state_mask1).item())
        has_mode2 = bool(torch.any(command_state_mask2).item())

        style_reward_mode0 = _safe_masked_mean(style_reward, command_state_mask0)
        style_reward_mode1 = _safe_masked_mean(style_reward, command_state_mask1)
        style_reward_mode2 = _safe_masked_mean(style_reward, command_state_mask2)
        style_reward_mode0_p50 = _safe_masked_quantile(style_reward, command_state_mask0, 0.5)
        style_reward_mode0_p90 = _safe_masked_quantile(style_reward, command_state_mask0, 0.9)
        style_reward_mode2_p50 = _safe_masked_quantile(style_reward, command_state_mask2, 0.5)
        style_reward_mode2_p90 = _safe_masked_quantile(style_reward, command_state_mask2, 0.9)

        amp_logits_view = amp_logits.reshape(rewards.shape)
        logits_mode0_mean = _safe_masked_mean(amp_logits_view, command_state_mask0)
        logits_mode0_std = _safe_masked_std(amp_logits_view, command_state_mask0)
        logits_mode2_mean = _safe_masked_mean(amp_logits_view, command_state_mask2)
        logits_mode2_std = _safe_masked_std(amp_logits_view, command_state_mask2)


        # 检查rewards和style_reward是否为NaN或Inf
        if torch.any(torch.isnan(rewards)) or torch.any(torch.isinf(rewards)):
            logger.warning("rewards包含NaN或Inf，使用nan_to_num修复")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.any(torch.isnan(style_reward)) or torch.any(torch.isinf(style_reward)):
            logger.warning("style_reward在combined前包含NaN或Inf，使用nan_to_num修复")
            style_reward = torch.nan_to_num(style_reward, nan=0.0, posinf=0.0, neginf=0.0)

        # 根据 command_state 使用不同的 style_reward_weight (mode0 vs mode2)
        # mode1 (站立) 的 style_reward 已被 walking_mask 置为0，此处权重不影响
        style_weight_per_sample = torch.where(
            command_state.unsqueeze(-1) == 2,
            torch.tensor(self._style_reward_weight_mode2, device=style_reward.device, dtype=style_reward.dtype),
            torch.tensor(self._style_reward_weight_mode0, device=style_reward.device, dtype=style_reward.dtype),
        )
        combined_rewards = self._task_reward_weight * rewards + style_weight_per_sample * style_reward

        # 检查combined_rewards是否为NaN或Inf
        if torch.any(torch.isnan(combined_rewards)) or torch.any(torch.isinf(combined_rewards)):
            logger.warning("combined_rewards包含NaN或Inf，使用nan_to_num修复")
            combined_rewards = torch.nan_to_num(combined_rewards, nan=0.0, posinf=0.0, neginf=0.0)



        # compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")

        # 检查values和next_values是否为NaN或Inf
        if torch.any(torch.isnan(values)) or torch.any(torch.isinf(values)):
            logger.warning("values包含NaN或Inf，使用nan_to_num修复")
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.any(torch.isnan(next_values)) or torch.any(torch.isinf(next_values)):
            logger.warning("next_values包含NaN或Inf，使用nan_to_num修复")
            next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)

        returns, advantages = compute_gae_group(
            rewards=combined_rewards,
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=next_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
            group_idx=command_state.unsqueeze(-1),
        )

        # 检查returns和advantages是否为NaN或Inf
        if torch.any(torch.isnan(returns)) or torch.any(torch.isinf(returns)):
            logger.warning("returns包含NaN或Inf，使用nan_to_num修复")
            returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
            logger.warning("advantages在GAE后包含NaN或Inf，使用nan_to_num修复")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)
        sampled_motion_batches = self.motion_dataset.sample(
            names=["states"], batch_size=self.memory.memory_size * self.memory.num_envs, mini_batches=self._mini_batches
        )
        if self._with_reply_sample:
            if len(self.reply_buffer):
                sampled_replay_batches = self.reply_buffer.sample(
                    names=["states"],
                    batch_size=self.memory.memory_size * self.memory.num_envs,
                    mini_batches=self._mini_batches,
                )
            else:
                sampled_replay_batches = [[batches[self.tensors_names.index("amp_states")]] for batches in sampled_batches]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0
        cumulative_discriminator_policy_loss = 0
        cumulative_discriminator_expert_loss = 0
        cumulative_discriminator_logit_regularization = 0
        cumulative_discriminator_gradient_panelty = 0
        cumulative_discriminator_weight_decay = 0
        cumulative_actor_sym_loss = 0
        cumulative_critic_sym_loss = 0 
        cumulative_discriminator_sym_loss = 0



        mean_style_reward = style_reward.mean().item()
        mean_task_reward = rewards.mean().item()
        weighted_task_reward_mean = (self._task_reward_weight * rewards).mean().item()
        weighted_style_reward_mean = (style_weight_per_sample * style_reward).mean().item()

        discriminator_loss = None
        discriminator_policy_loss = None

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch_index, (
                sampled_states,
                sampled_observations,
                sampled_actions,
                _,
                _,
                _,
                _,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_amp_states,
                _,
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    
                    
                    if self._use_symmetry_loss:
                        flip_state = self.flip_critic_obs(sampled_states)
                        flip_sampled_policy_observation = self.flip_actor_obs(sampled_observations)
                        flip_sampled_amp_states = self.flip_amp_obs(sampled_amp_states)
                        flip_sampled_motion_states = self.flip_amp_obs(sampled_motion_batches[batch_index][0])


                    sampled_obs_dim = sampled_observations.shape[1]
                    sampled_has_command_state = sampled_obs_dim not in (72, 51)
                    if sampled_has_command_state:
                        sampled_command_state = sampled_observations[:, 9]
                    else:
                        sampled_command_state = torch.zeros_like(sampled_observations[:, 0])
                    walking_mask = ((sampled_command_state == 0) | (sampled_command_state == 2)).detach()
                    lower_body_only_mask = (sampled_command_state == 2).detach()

                    sampled_states = self._state_preprocessor(sampled_states, train=True)
                    sampled_observations = self._observation_preprocessor(sampled_observations, train=True)

                    if self._use_symmetry_loss:
                        flip_state = self._state_preprocessor(flip_state, train=True)
                        flip_sampled_policy_observation = self._observation_preprocessor(flip_sampled_policy_observation, train=True)
                    
                    
                    _, next_log_prob, _ = self.policy.act(
                        {"observations": sampled_observations, "states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"observations": sampled_observations, "states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # compute discriminator loss
                    if self._discriminator_batch_size:
                        walking_indices = torch.where(walking_mask)[0]
                        if len(walking_indices) > 0:
                            batch_size = min(self._discriminator_batch_size, len(walking_indices))
                            filtered_walking_indices = walking_indices[:batch_size]
                            lower_body_only_mask_filtered = lower_body_only_mask[filtered_walking_indices]
                        else:
                            lower_body_only_mask_filtered = torch.zeros(0, dtype=torch.bool, device=walking_mask.device)
                    else:
                        walking_indices = torch.where(walking_mask)[0]
                        lower_body_only_mask_filtered = lower_body_only_mask[walking_indices]

                    if self._discriminator_batch_size:
                        sampled_amp_states = self.scale_amp_obs(self._amp_state_preprocessor(
                            sampled_amp_states[walking_mask][0 : self._discriminator_batch_size], train=True
                        ))
                        if self._use_symmetry_loss:
                            flip_sampled_amp_states = self.scale_amp_obs(self._amp_state_preprocessor(
                                flip_sampled_amp_states[walking_mask][0 : self._discriminator_batch_size]
                            , train=True))
                        if self._with_reply_sample:
                            sampled_amp_replay_states = self.scale_amp_obs(self._amp_state_preprocessor(
                                sampled_replay_batches[batch_index][0][walking_mask][0 : self._discriminator_batch_size], train=True
                        ))
                        sampled_amp_motion_states = self.scale_amp_obs(self._amp_state_preprocessor(
                            sampled_motion_batches[batch_index][0][walking_mask][0 : self._discriminator_batch_size], train=True
                        ))
                        if self._use_symmetry_loss:
                            flip_motion_amp_states = self.scale_amp_obs(self._amp_state_preprocessor(
                               flip_sampled_motion_states[walking_mask][0 : self._discriminator_batch_size]
                            , train=True))
                    else:
                        sampled_amp_states = self.scale_amp_obs(self._amp_state_preprocessor(sampled_amp_states[walking_mask], train=True))
                        if self._with_reply_sample:

                            sampled_amp_replay_states = self.scale_amp_obs(self._amp_state_preprocessor(
                                sampled_replay_batches[batch_index][0][walking_mask], train=True
                            ))
                        sampled_amp_motion_states = self.scale_amp_obs(self._amp_state_preprocessor(
                            sampled_motion_batches[batch_index][0][walking_mask], train=True
                        ))

                    # 当command_state == 2时，mask掉上半身特征（只关注下半身）
                    # state=0: 全身，不应用mask；state=2: 只下半身，mask掉上半身特征
                    # RobanS2(21关节,64dim): dof_pos[13:21], dof_vel[34:42], key_body_pos[52:58] (手部)
                    # if len(lower_body_only_mask_filtered) > 0 and lower_body_only_mask_filtered.any():
                    #     # Convert (lab -> gym) before applying masking indices which are defined in gym order
                    #     sampled_amp_states = self._amp_states_lab_to_gym(sampled_amp_states)
                    #     sampled_amp_motion_states = self._amp_states_lab_to_gym(sampled_amp_motion_states)
                    #     if self._with_reply_sample:
                    #         sampled_amp_replay_states = self._amp_states_lab_to_gym(sampled_amp_replay_states)
                    #     if self._use_symmetry_loss:
                    #         flip_sampled_amp_states = self._amp_states_lab_to_gym(flip_sampled_amp_states)
                    #         flip_motion_amp_states = self._amp_states_lab_to_gym(flip_motion_amp_states)
                    #
                    #     single_amp_obs = self.amp_observation_space.shape[0] // self._discriminator_history_length
                    #     dof_pos_sl, dof_vel_sl, key_body_sl = self._get_upper_body_mask_slices(single_amp_obs)
                    #     for i in range(self._discriminator_history_length):
                    #         start_idx = i * single_amp_obs
                    #         # mask掉上半身关节位置
                    #         sampled_amp_states[lower_body_only_mask_filtered, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
                    #         sampled_amp_motion_states[lower_body_only_mask_filtered, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
                    #         # mask掉上半身关节速度
                    #         sampled_amp_states[lower_body_only_mask_filtered, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
                    #         sampled_amp_motion_states[lower_body_only_mask_filtered, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
                    #         # mask掉上半身key body位置（手部）
                    #         sampled_amp_states[lower_body_only_mask_filtered, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
                    #         sampled_amp_motion_states[lower_body_only_mask_filtered, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
                    #         if self._with_reply_sample:
                    #             sampled_amp_replay_states[lower_body_only_mask_filtered, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
                    #             sampled_amp_replay_states[lower_body_only_mask_filtered, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
                    #             sampled_amp_replay_states[lower_body_only_mask_filtered, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
                    #         if self._use_symmetry_loss:
                    #             flip_sampled_amp_states[lower_body_only_mask_filtered, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
                    #             flip_sampled_amp_states[lower_body_only_mask_filtered, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
                    #             flip_sampled_amp_states[lower_body_only_mask_filtered, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
                    #             flip_motion_amp_states[lower_body_only_mask_filtered, start_idx + dof_pos_sl.start:start_idx + dof_pos_sl.stop] = 0
                    #             flip_motion_amp_states[lower_body_only_mask_filtered, start_idx + dof_vel_sl.start:start_idx + dof_vel_sl.stop] = 0
                    #             flip_motion_amp_states[lower_body_only_mask_filtered, start_idx + key_body_sl.start:start_idx + key_body_sl.stop] = 0
                    #
                    #     # Convert back (gym -> lab) to keep discriminator inputs consistent with env AMP observation order
                    #     sampled_amp_states = self._amp_states_gym_to_lab(sampled_amp_states)
                    # #     sampled_amp_motion_states = self._amp_states_gym_to_lab(sampled_amp_motion_states)
                    #     if self._with_reply_sample:
                    #         sampled_amp_replay_states = self._amp_states_gym_to_lab(sampled_amp_replay_states)
                    #     if self._use_symmetry_loss:
                    #         flip_sampled_amp_states = self._amp_states_gym_to_lab(flip_sampled_amp_states)
                    #         flip_motion_amp_states = self._amp_states_gym_to_lab(flip_motion_amp_states)

                    # discriminator prediction loss
                    # discriminator_loss = 0.5 * (
                    #     nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
                    #     + torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
                    # )
                      # detach避免mask产生梯度

                    match self._discriminator_loss_type:
                        
                        case 'Wasserstein':

                            amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
                            amp_logits = torch.nan_to_num(amp_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            
                            amp_motion_logits, _, _ = self.discriminator.act(
                                {"states": sampled_amp_motion_states}, role="discriminator"
                            )
                            amp_motion_logits = torch.nan_to_num(amp_motion_logits, nan=0.0, posinf=50.0, neginf=-50.0)

                            random_tensor= torch.rand((1),device=sampled_amp_states.device)
                            random_factor_states=random_tensor*sampled_amp_states+(1-random_tensor)*sampled_amp_motion_states
                            random_factor_states.requires_grad_(True)
                            amp_motion_mix, _, _ = self.discriminator.act(
                                {"states": random_factor_states}, role="discriminator"
                            )


                            discriminator_expert_loss =torch.tanh(self._soft_boundary_constraint_scale*amp_motion_logits).mean()
                            discriminator_policy_loss =torch.tanh(self._soft_boundary_constraint_scale*amp_logits).mean()
                            discriminator_loss = (discriminator_policy_loss-discriminator_expert_loss)

                            # discriminator gradient penalty
                            if self._discriminator_gradient_penalty_scale:
                                amp_motion_gradient = torch.autograd.grad(
                                    amp_motion_mix,
                                    random_factor_states,
                                    grad_outputs=torch.ones_like(amp_motion_mix),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                )
                                gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                                discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty
                        
                        case 'BCE':
                            sampled_amp_motion_states.requires_grad_(True)
                            amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
                            amp_logits = torch.nan_to_num(amp_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            if self._with_reply_sample:
                                amp_replay_logits, _, _ = self.discriminator.act(
                                    {"states": sampled_amp_replay_states}, role="discriminator"
                                )
                                amp_replay_logits = torch.nan_to_num(amp_replay_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            amp_motion_logits, _, _ = self.discriminator.act(
                                {"states": sampled_amp_motion_states}, role="discriminator"
                            )
                            amp_motion_logits = torch.nan_to_num(amp_motion_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            amp_input_logits = amp_logits
                            if self._with_reply_sample:
                                amp_input_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)
                            
                            
                            discriminator_policy_loss = nn.BCEWithLogitsLoss()(amp_input_logits, torch.zeros_like(amp_input_logits))
                            discriminator_expert_loss = torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
                            discriminator_loss = 0.5*(discriminator_policy_loss+discriminator_expert_loss)
                            

                            # discriminator gradient penalty
                            if self._discriminator_gradient_penalty_scale:
                                amp_motion_gradient = torch.autograd.grad(
                                    amp_motion_logits,
                                    sampled_amp_motion_states,
                                    grad_outputs=torch.ones_like(amp_motion_logits),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                )
                                gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                                discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty
                        case 'MSE':
                            sampled_amp_motion_states.requires_grad_(True)
                            amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
                            amp_logits = torch.nan_to_num(amp_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            if self._with_reply_sample:
                                amp_replay_logits, _, _ = self.discriminator.act(
                                    {"states": sampled_amp_replay_states}, role="discriminator"
                                )
                                amp_replay_logits = torch.nan_to_num(amp_replay_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            amp_motion_logits, _, _ = self.discriminator.act(
                                {"states": sampled_amp_motion_states}, role="discriminator"
                            )
                            amp_motion_logits = torch.nan_to_num(amp_motion_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            amp_input_logits = amp_logits
                            if self._with_reply_sample:
                                amp_input_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)
                            
                            
                            discriminator_expert_loss = torch.nn.MSELoss()(
                                amp_motion_logits, torch.ones(amp_motion_logits.size(), device=self.device)
                            )
                            discriminator_policy_loss = torch.nn.MSELoss()(
                                amp_input_logits, -1 * torch.ones(amp_input_logits.size(), device=self.device)
                            )
                            discriminator_loss = 0.5 * (discriminator_expert_loss + discriminator_policy_loss)

                            # discriminator gradient penalty
                            if self._discriminator_gradient_penalty_scale:
                                amp_motion_gradient = torch.autograd.grad(
                                    amp_motion_logits,
                                    sampled_amp_motion_states,
                                    grad_outputs=torch.ones_like(amp_motion_logits),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                )
                                gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                                discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty


                        # discriminator logit regularization
                    if self._discriminator_logit_regularization_scale:
                        logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                        # discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                        #     torch.square(logit_weights)
                        # )
                        discriminator_logit_regularization = torch.sum(torch.square(logit_weights))
                        discriminator_loss += self._discriminator_logit_regularization_scale * discriminator_logit_regularization

                   
                    

                    # discriminator weight decay
                    if self._discriminator_weight_decay_scale:
                        weights = [
                            torch.flatten(module.weight)
                            for module in self.discriminator.modules()
                            if isinstance(module, torch.nn.Linear)
                        ]
                        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                        discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

                    discriminator_loss *= self._discriminator_loss_scale



                    if self._use_symmetry_loss:
                        
                        action_flip = self.policy.act({"observations":flip_sampled_policy_observation} , role="policy")[2]["mean_actions"]
                        infer_action = self.policy.act({"observations":sampled_observations}, role="policy")[2]["mean_actions"]
                        flip_action_src = self.flip_action(infer_action)
                        actor_sym_loss = self._symmetry_scale * torch.mean(torch.sum(torch.square(action_flip - flip_action_src), dim=-1))
                        
                        predicted_values_sym = self.value.act({"observations": flip_sampled_policy_observation, "states": flip_state}, role="value")[0]
                        predicted_values_src = self.value.act({"observations": sampled_observations, "states": sampled_states}, role="value")[0]
                        critic_sym_loss = self._symmetry_scale * torch.mean(torch.square(predicted_values_sym - predicted_values_src.detach()))
                        
                        
                        if self._disc_sym_loss_mode==3:
                            flip_amp_policy_loss=   torch.tanh(self.discriminator.act({"states": flip_sampled_amp_states}, role="discriminator")[0]).mean()
                            amp_policy_loss =torch.tanh(self.discriminator.act({"states": sampled_amp_states}, role="discriminator")[0]).mean()
                            flip_amp_motion_loss=   torch.tanh(self.discriminator.act({"states": flip_motion_amp_states}, role="discriminator")[0]).mean()
                            amp_motion_src_loss =torch.tanh(self.discriminator.act({"states": sampled_amp_motion_states}, role="discriminator")[0]).mean()
                            discriminator_sym_loss = self._symmetry_scale * (torch.mean(torch.square(flip_amp_policy_loss-amp_policy_loss.detach()))+
                                                                        torch.mean(torch.square(flip_amp_motion_loss-amp_motion_src_loss.detach())))
                        elif self._disc_sym_loss_mode==2:
                            flip_amp_motion_loss=   torch.tanh(self.discriminator.act({"states": flip_motion_amp_states}, role="discriminator")[0]).mean()
                            amp_motion_src_loss =torch.tanh(self.discriminator.act({"states": sampled_amp_motion_states}, role="discriminator")[0]).mean()
                            discriminator_sym_loss = self._symmetry_scale * torch.mean(torch.square(flip_amp_motion_loss-amp_motion_src_loss.detach()))
                                                                        
                        elif self._disc_sym_loss_mode==1:
                            flip_amp_policy_loss=   torch.tanh(self.discriminator.act({"states": flip_sampled_amp_states}, role="discriminator")[0]).mean()
                            amp_policy_loss =torch.tanh(self.discriminator.act({"states": sampled_amp_states}, role="discriminator")[0]).mean()
                            discriminator_sym_loss = self._symmetry_scale * torch.mean(torch.square(flip_amp_policy_loss-amp_policy_loss.detach()))
                        else:# 0 or 4
                            discriminator_sym_loss = torch.zeros_like(critic_sym_loss,device=critic_sym_loss.device)
                        
                # optimization step
                # 修改：分别对策略/价值网络和判别器进行优化
                self.optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                # 当站立或下蹲弯腰时,去掉discriminator的奖励与LOSS
                # 创建行走任务的mask (只有行走时才使用discriminator loss)
                # walking_mask = (sampled_observations[:, 9] <= 0).float().detach()  # detach避免mask产生梯度
                # discriminator_loss = (discriminator_loss * walking_mask).mean()
                # discriminator_sym_loss = (discriminator_sym_loss * walking_mask).mean()

                # 修改：分别计算策略/价值网络和判别器的损失
                if self._use_symmetry_loss:
                    # 分别缩放和反向传播策略/价值网络损失和判别器损失
                    self.scaler.scale(policy_loss + entropy_loss + value_loss + actor_sym_loss + critic_sym_loss).backward(retain_graph=True)
                    self.scaler.scale(discriminator_loss + discriminator_sym_loss).backward()
                else:
                    # 分别缩放和反向传播策略/价值网络损失和判别器损失
                    self.scaler.scale(policy_loss + entropy_loss + value_loss).backward(retain_graph=True)
                    self.scaler.scale(discriminator_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    self.value.reduce_parameters()
                    self.discriminator.reduce_parameters()

                # 修改：分别对策略/价值网络和判别器进行梯度裁剪
                # 警告：如果grad_norm_clip为0，梯度可能会爆炸导致NaN
                if self._grad_norm_clip > 0:
                    # 对策略/价值网络进行梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    try:
                        grad_norm = nn.utils.clip_grad_norm_(
                            itertools.chain(
                                self.policy.parameters(), self.value.parameters()
                            ),
                            self._grad_norm_clip,
                        )
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(f"策略/价值网络梯度范数为NaN/Inf: {grad_norm}")
                    except RuntimeError as e:
                        logger.error(f"策略/价值网络梯度裁剪失败: {e}")

                    # 对判别器进行梯度裁剪
                    self.scaler.unscale_(self.discriminator_optimizer)
                    try:
                        disc_grad_norm = nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(),
                            self._grad_norm_clip,
                        )
                        if torch.isnan(disc_grad_norm) or torch.isinf(disc_grad_norm):
                            logger.warning(f"判别器梯度范数为NaN/Inf: {disc_grad_norm}")
                    except RuntimeError as e:
                        logger.error(f"判别器梯度裁剪失败: {e}")
                else:
                    # 如果没有梯度裁剪，检查梯度是否为NaN或Inf
                    self.scaler.unscale_(self.optimizer)
                    for name, param in itertools.chain(self.policy.named_parameters(), self.value.named_parameters()):
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                logger.warning(f"策略/价值网络参数 {name} 的梯度包含NaN/Inf，建议设置grad_norm_clip > 0")
                                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                    self.scaler.unscale_(self.discriminator_optimizer)
                    for name, param in self.discriminator.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                logger.warning(f"判别器参数 {name} 的梯度包含NaN/Inf，建议设置grad_norm_clip > 0")
                                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

                # 修改：分别对策略/价值网络和判别器进行优化步骤
                self.scaler.step(self.optimizer)
                self.scaler.step(self.discriminator_optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()
                cumulative_discriminator_policy_loss += discriminator_policy_loss.item()
                cumulative_discriminator_expert_loss += discriminator_expert_loss.item()
                cumulative_discriminator_logit_regularization += discriminator_logit_regularization.item()
                cumulative_discriminator_gradient_panelty += gradient_penalty.item()
                cumulative_discriminator_weight_decay += weight_decay.item()
                if self._use_symmetry_loss:
                    cumulative_actor_sym_loss +=actor_sym_loss.item()
                    cumulative_critic_sym_loss +=critic_sym_loss.item()
                    cumulative_discriminator_sym_loss +=discriminator_sym_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                    # 新增：更新判别器的学习率调度器
                    self.discriminator_scheduler.step(kl.item())
                else:
                    self.scheduler.step()
                    # 新增：更新判别器的学习率调度器
                    self.discriminator_scheduler.step()

        # update AMP replay buffer
        if self._with_reply_sample:
            self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1])) 

        if self._use_symmetry_loss:
            self.track_data("Loss / Policy sym loss", cumulative_actor_sym_loss / (self._learning_epochs * self._mini_batches))
            self.track_data("Loss / Value sym loss", cumulative_critic_sym_loss / (self._learning_epochs * self._mini_batches))
            self.track_data("Loss / discriminator sym loss", cumulative_discriminator_sym_loss / (self._learning_epochs * self._mini_batches))

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )
        self.track_data(
            "Loss / Discriminator loss", cumulative_discriminator_loss / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Discriminator policy loss", cumulative_discriminator_policy_loss / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Discriminator expert loss", cumulative_discriminator_expert_loss / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Discriminator logit regularization", cumulative_discriminator_logit_regularization / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Discriminator gradient panelty", cumulative_discriminator_gradient_panelty / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Discriminator weight decay", cumulative_discriminator_weight_decay / (self._learning_epochs * self._mini_batches)
        )

        # 检查policy stddev是否为NaN
        policy_stddev = self.policy.distribution(role="policy").stddev
        if torch.any(torch.isnan(policy_stddev)) or torch.any(torch.isinf(policy_stddev)):
            logger.warning("policy stddev包含NaN或Inf")
            policy_stddev_mean = 0.0
        else:
            policy_stddev_mean = policy_stddev.mean().item()
        self.track_data("Policy / Standard deviation", policy_stddev_mean)

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
            # 新增：记录判别器的学习率
            self.track_data("Learning / Discriminator learning rate", self.discriminator_scheduler.get_last_lr()[0])

        self.track_data("Reward / Style instantaneous reward (mean)", mean_style_reward)
        self.track_data("Reward / Task reward (mean)", mean_task_reward)
        self.track_data("Reward / Weighted style reward (mean)", weighted_style_reward_mean)
        self.track_data("Reward / Weighted task reward (mean)", weighted_task_reward_mean)
        if has_mode0:
            self.track_data("Reward / Style reward mean (mode 0)", style_reward_mode0)
            self.track_data("Reward / Style reward p50 (mode 0)", style_reward_mode0_p50)
            self.track_data("Reward / Style reward p90 (mode 0)", style_reward_mode0_p90)
            self.track_data("Discriminator / Logits mean (mode 0)", logits_mode0_mean)
            self.track_data("Discriminator / Logits std (mode 0)", logits_mode0_std)
        if has_mode1:
            self.track_data("Reward / Style reward mean (mode 1)", style_reward_mode1)
        if has_mode2:
            self.track_data("Reward / Style reward mean (mode 2)", style_reward_mode2)
            self.track_data("Reward / Style reward p50 (mode 2)", style_reward_mode2_p50)
            self.track_data("Reward / Style reward p90 (mode 2)", style_reward_mode2_p90)
            self.track_data("Discriminator / Logits mean (mode 2)", logits_mode2_mean)
            self.track_data("Discriminator / Logits std (mode 2)", logits_mode2_std)



    def _get_upper_body_mask_slices(self, single_amp_obs: int):
        """根据单帧 AMP 观测维度返回上半身遮蔽的切片范围。

        RobanS2 (21关节, single_amp_obs=64):
          dof布局: dof_pos(21) | dof_vel(21) | base_z(1) | proj_grav(3) | lin_vel(3) | ang_vel(3) | key_body_pos(12)
          上半身 = 手臂关节 (lab索引 13-20):
            dof_pos_slice=[13:21], dof_vel_slice=[34:42], key_body_slice=[52:58] (右手+左手)

        """
        if single_amp_obs >= 70: 
            return slice(12, 27), slice(39, 54), slice(64, 70)
        else:  
            return slice(13, 21), slice(34, 42), slice(52, 58)

    def _amp_states_lab_to_gym(self, amp_states: torch.Tensor) -> torch.Tensor:
        """Reorder AMP dof_pos/dof_vel from lab order -> gym order for each discriminator history frame.

        Only the first (n_dof * 2) dims of each frame are reordered.
        The rest (root / key-body related features) are left unchanged.

        gym indices are used by the masking logic in this agent.
        If gym order == lab order (identity mapping), this is effectively a no-op.
        """
        if amp_states.numel() == 0:
            return amp_states
        n_dof = self._amp_dof_gym2lab.numel()
        if n_dof == 0:
            return amp_states
        # last dim is (history_len * single_amp_obs)
        total_dim = amp_states.shape[-1]
        if total_dim % self._discriminator_history_length != 0:
            return amp_states
        single_amp_obs = total_dim // self._discriminator_history_length
        if single_amp_obs < n_dof * 2:
            return amp_states

        # lab -> gym: output[gym] = input[lab_index_of_gym] == input[gym2lab[gym]]
        idx = self._amp_dof_gym2lab.to(device=amp_states.device)
        amp_view = amp_states.view(*amp_states.shape[:-1], self._discriminator_history_length, single_amp_obs)
        out = amp_view.clone()
        out[..., :n_dof] = amp_view[..., :n_dof].index_select(-1, idx)
        out[..., n_dof:n_dof * 2] = amp_view[..., n_dof:n_dof * 2].index_select(-1, idx)
        return out.view(*amp_states.shape)

    def _amp_states_gym_to_lab(self, amp_states: torch.Tensor) -> torch.Tensor:
        """Reorder AMP dof_pos/dof_vel from gym order -> lab order for each discriminator history frame.

        If gym order == lab order (identity mapping), this is effectively a no-op.
        """
        if amp_states.numel() == 0:
            return amp_states
        n_dof = self._amp_dof_lab2gym.numel()
        if n_dof == 0:
            return amp_states
        total_dim = amp_states.shape[-1]
        if total_dim % self._discriminator_history_length != 0:
            return amp_states
        single_amp_obs = total_dim // self._discriminator_history_length
        if single_amp_obs < n_dof * 2:
            return amp_states

        # gym -> lab: output[lab] = input[gym_index_of_lab] == input[lab2gym[lab]]
        idx = self._amp_dof_lab2gym.to(device=amp_states.device)
        amp_view = amp_states.view(*amp_states.shape[:-1], self._discriminator_history_length, single_amp_obs)
        out = amp_view.clone()
        out[..., :n_dof] = amp_view[..., :n_dof].index_select(-1, idx)
        out[..., n_dof:n_dof * 2] = amp_view[..., n_dof:n_dof * 2].index_select(-1, idx)
        return out.view(*amp_states.shape)

    def flip_action(self, actions):
        """
        Universal action flipping method that automatically detects robot type.
        """
        num_joints = actions.shape[-1]
        
        if num_joints == 21:
            return self.flip_roban_action(actions)
        else:
            raise ValueError(f"Unsupported action dimension: {num_joints}. Expected  21 (Roban)")

    def flip_amp_obs(self, amp_obs):
        """
        Universal AMP observation flipping method that automatically detects robot type.
        Supports Roban (64 dims per frame).
        amp:     ['joint_pos', 'joint_vel', 'base_pos_z', 'projected_gravity', 'root_lin_vel', 'root_ang_vel', 'rel_key_body_pos']
        wbc amp: ['joint_pos', 'joint_vel', 'base_pos_z', 'projected_gravity', 'root_lin_vel', 'root_ang_vel', 'rel_key_body_pos']
        """
        # Get single frame dimension
        single_frame_dim = self.amp_observation_space.shape[0] // self._discriminator_history_length
        
        # Roban: 21+21+1+3+3+3+12 = 64
        # print("==flip_amp_obs==========================================")
        # print("single_frame_dim:", single_frame_dim)
        # print("---------------------------------------------------------")
        if single_frame_dim == 64 or single_frame_dim <= 65:
            # Roban robot
            return self.flip_roban_amp_obs(amp_obs)
        else:
            raise ValueError(f"Unsupported AMP observation dimension: {single_frame_dim}. Expected 64 (Roban)")

    def scale_amp_obs(self, amp_obs):
        """
        Universal AMP observation scaling method that automatically detects robot type.
        Scales hip_pitch, knee, ankle_pitch joints (sagittal plane joints).
        Supports Roban (21 joints).
        """
        single_amp_obs = self.amp_observation_space.shape[0] // self._discriminator_history_length
        
        # Roban: 64 dims per frame
        # Squat : roban21 =64 ---checked 
        if single_amp_obs == 64 or single_amp_obs <= 65:
            # Roban robot
            return self.scale_roban_amp_obs(amp_obs)
        else:
            raise ValueError(f"Unsupported AMP observation dimension: {single_amp_obs}. Expected 64 (Roban)")

    def flip_critic_obs(self, critic_obs):
        """
        Universal critic observation flipping method that automatically detects robot type.
        Supports both Roban (150 dims).
        """
        obs_dim = self.state_space.shape[0]
        # print ("==flip_critic_obs=================================================")
        # print ("critic_obs_dim for robot :",obs_dim)

        # print ("---------------------------------------------------------------")

        # Roban: 3+3+3+21+21+21+3+21+21+6+6+1+6+3+1+3+3+2+2 = 150 + 1(cmdst) +21 joint_erro+ 1(height_error) +1(bending_error) =174
        if obs_dim >= 165 and obs_dim <= 190:
            # Roban robot (165 = 167-style layout without height/bending; 167–174 see flip_roban_critic_obs)
            return self.flip_roban_critic_obs(critic_obs)
        else:
            # Some experiments prune critic observation terms (e.g. remove push/height/bending terms).
            # In that case, skip symmetry flip instead of indexing out of bounds in fixed-layout mappers.
            return critic_obs


    def flip_actor_obs(self, obs):
        """
        Universal actor observation flipping method that automatically detects robot type.
        Supports Roban (~48 dims) without actions, or with actions.
        policy:['base_ang_vel', 'projected_gravity', 'velocity_commands', 
        'command_state'  --add for squat
        'joint_pos', 'joint_vel', 'actions']
        """
        obs_dim = self.observation_space.shape[0]
        
        # Roban without actions: 3+3+3+ + 1+ 21+21 = 52
        # Roban with actions: 52+21 = 73   --checked
        # print("==flip_actor_obs=================================================")
        # print("actor_obs_dim for robot :",obs_dim)
        # print("-----------------------------------------------------------------")
        if obs_dim >= 60 and obs_dim < 80:
            if obs_dim >= 70:
                # Roban with actions
                return self.flip_roban_actor_obs(obs)
        elif obs_dim >= 48 and obs_dim < 60:
            # Roban robot (without actions)
            return self.flip_roban_actor_obs(obs)
        else:
            raise ValueError(f"Unsupported actor observation dimension: {obs_dim}")

    
    def flip_roban_action(self, actions):
        flipped_actions = torch.zeros_like(actions)
        
        flipped_actions[:,  0] = -actions[:, 0]        # 0 "waist_yaw_joint",   
        flipped_actions[:,  1] = -actions[:, 7]        # 1 "left_hip_roll_joint"
        flipped_actions[:,  2] = -actions[:, 8]        # 2 "left_hip_yaw_joint",
        flipped_actions[:,  3] = -actions[:, 9]        # 3 "left_hip_pitch_joint",
        flipped_actions[:,  4] =  actions[:, 10]       # 4 "left_knee_joint",
        flipped_actions[:,  5] =  actions[:, 11]       # 5 "left_ankle_pitch_joint"
        flipped_actions[:,  6] = -actions[:, 12]       # 6 "left_ankle_roll_joint",
      
        flipped_actions[:,  7] = -actions[:, 1]        # 7 "right_hip_roll_joint"
        flipped_actions[:,  8] = -actions[:, 2]        # 8 "right_hip_yaw_joint",
        flipped_actions[:,  9] = -actions[:, 3]        # 9 "right_hip_pitch_joint",
        flipped_actions[:, 10] =  actions[:, 4]        # 10 "right_knee_joint",
        flipped_actions[:, 11] =  actions[:, 5]        # 11 "right_ankle_pitch_joint"
        flipped_actions[:, 12] = -actions[:, 6]        # 12 "right_ankle_roll_joint",
        
        flipped_actions[:, 13] =  actions[:, 17]       
        flipped_actions[:, 14] = -actions[:, 18]       
        flipped_actions[:, 15] = -actions[:, 19]       
        flipped_actions[:, 16] =  actions[:, 20]    
             
        flipped_actions[:, 17] =  actions[:, 13]       
        flipped_actions[:, 18] = -actions[:, 14]       
        flipped_actions[:, 19] = -actions[:, 15]       
        flipped_actions[:, 20] =  actions[:, 16]  
        return flipped_actions.detach()

    def flip_roban_amp_obs(self, amp_obs):
        """
        Flip roban amp observations for symmetry
        roban amp:['joint_pos', 'joint_vel', 'base_pos_z', 'projected_gravity', 'root_lin_vel', 'root_ang_vel', 'rel_key_body_pos']
        Roban: 21 joints, AMP obs dimension = 21+21+1+3+3+3+12 = 64
        """
        proprioceptive_obs = torch.clone(amp_obs[:, :self.amp_observation_space.shape[0]])
        proprioceptive_obs = proprioceptive_obs.view(-1, self._discriminator_history_length, self.amp_observation_space.shape[0]//self._discriminator_history_length)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        # Joint positions (0-20): 21 joints
        flipped_proprioceptive_obs[:, :, 0] = -proprioceptive_obs[:, :, 0]   # waist_yaw取负
        
        # Left leg -> Right leg
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 7]   # left_hip_roll -> right_hip_roll (取负)
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 8]   # left_hip_yaw -> right_hip_yaw
        flipped_proprioceptive_obs[:, :, 3] = -proprioceptive_obs[:, :, 9]   # left_hip_pitch -> right_hip_pitch
        flipped_proprioceptive_obs[:, :, 4] =  proprioceptive_obs[:, :, 10]  # left_knee -> right_knee
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 11]  # left_ankle_pitch -> right_ankle_pitch (取负)
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 12]  # left_ankle_roll -> right_ankle_roll (取负)

        # Right leg -> Left leg
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 1]   # right_hip_roll -> left_hip_roll (取负)
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 2]   # right_hip_yaw -> left_hip_yaw
        flipped_proprioceptive_obs[:, :, 9] = -proprioceptive_obs[:, :, 3]   # right_hip_pitch -> left_hip_pitch
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 4]  # right_knee -> left_knee
        flipped_proprioceptive_obs[:, :, 11] =  proprioceptive_obs[:, :, 5]  # right_ankle_pitch -> left_ankle_pitch (取负)
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 6]  # right_ankle_roll -> left_ankle_roll (取负)
        
        # Left arm -> Right arm
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 17]  # left_shoulder_pitch -> right_shoulder_pitch (取负)
        flipped_proprioceptive_obs[:, :, 14] = -proprioceptive_obs[:, :, 18]  # left_shoulder_roll -> right_shoulder_roll (取负)
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 19]  # left_shoulder_yaw -> right_shoulder_yaw
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 20]  # left_elbow -> right_elbow (取负)
        
        # Right arm -> Left arm
        flipped_proprioceptive_obs[:, :, 17] =  proprioceptive_obs[:, :, 13]  # right_shoulder_pitch -> left_shoulder_pitch
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 14]  # right_shoulder_roll -> left_shoulder_roll (取负)
        flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 15]  # right_shoulder_yaw -> left_shoulder_yaw (取负)
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 16]  # right_elbow -> left_elbow

        # Joint velocities (21-41): 21 joints
        flipped_proprioceptive_obs[:, :, 0+21] = -proprioceptive_obs[:, :, 0+21]   # waist_yaw_vel取负
        
        # Left leg vel -> Right leg vel
        flipped_proprioceptive_obs[:, :, 1+21] = -proprioceptive_obs[:, :, 7+21]
        flipped_proprioceptive_obs[:, :, 2+21] = -proprioceptive_obs[:, :, 8+21]
        flipped_proprioceptive_obs[:, :, 3+21] = -proprioceptive_obs[:, :, 9+21]
        flipped_proprioceptive_obs[:, :, 4+21] =  proprioceptive_obs[:, :, 10+21]
        flipped_proprioceptive_obs[:, :, 5+21] =  proprioceptive_obs[:, :, 11+21]
        flipped_proprioceptive_obs[:, :, 6+21] = -proprioceptive_obs[:, :, 12+21]

        # Right leg vel -> Left leg vel
        flipped_proprioceptive_obs[:, :, 7+21] = -proprioceptive_obs[:, :, 1+21]
        flipped_proprioceptive_obs[:, :, 8+21] = -proprioceptive_obs[:, :, 2+21]
        flipped_proprioceptive_obs[:, :, 9+21] = -proprioceptive_obs[:, :, 3+21]
        flipped_proprioceptive_obs[:, :, 10+21] =  proprioceptive_obs[:, :, 4+21]
        flipped_proprioceptive_obs[:, :, 11+21] =  proprioceptive_obs[:, :, 5+21]
        flipped_proprioceptive_obs[:, :, 12+21] = -proprioceptive_obs[:, :, 6+21]
        
        # Left arm vel -> Right arm vel
        flipped_proprioceptive_obs[:, :, 13+21] =  proprioceptive_obs[:, :, 17+21]
        flipped_proprioceptive_obs[:, :, 14+21] = -proprioceptive_obs[:, :, 18+21]
        flipped_proprioceptive_obs[:, :, 15+21] = -proprioceptive_obs[:, :, 19+21]
        flipped_proprioceptive_obs[:, :, 16+21] =  proprioceptive_obs[:, :, 20+21]
        
        # Right arm vel -> Left arm vel
        flipped_proprioceptive_obs[:, :, 17+21] =  proprioceptive_obs[:, :, 13+21]
        flipped_proprioceptive_obs[:, :, 18+21] = -proprioceptive_obs[:, :, 14+21]
        flipped_proprioceptive_obs[:, :, 19+21] = -proprioceptive_obs[:, :, 15+21]
        flipped_proprioceptive_obs[:, :, 20+21] =  proprioceptive_obs[:, :, 16+21]

        # base_pos_z (42): keep as is
        flipped_proprioceptive_obs[:, :, 42] =  proprioceptive_obs[:, :, 42]

        # projected_gravity (43-45): y方向取负
        flipped_proprioceptive_obs[:, :, 43] =  proprioceptive_obs[:, :, 43]
        flipped_proprioceptive_obs[:, :, 44] = -proprioceptive_obs[:, :, 44]
        flipped_proprioceptive_obs[:, :, 45] =  proprioceptive_obs[:, :, 45]
        
        # root_lin_vel (46-48): y方向取负
        flipped_proprioceptive_obs[:, :, 46] =  proprioceptive_obs[:, :, 46]
        flipped_proprioceptive_obs[:, :, 47] = -proprioceptive_obs[:, :, 47]
        flipped_proprioceptive_obs[:, :, 48] =  proprioceptive_obs[:, :, 48]
        
        # root_ang_vel (49-51): x和z方向取负
        flipped_proprioceptive_obs[:, :, 49] = -proprioceptive_obs[:, :, 49]
        flipped_proprioceptive_obs[:, :, 50] =  proprioceptive_obs[:, :, 50]
        flipped_proprioceptive_obs[:, :, 51] = -proprioceptive_obs[:, :, 51]
        
        # rel_key_body_pos (52-63): 12维，左右手、左右脚交换
        # Right hand (52-54) -> Left hand (55-57)
        flipped_proprioceptive_obs[:, :, 52] =  proprioceptive_obs[:, :, 55]  # right_hand
        flipped_proprioceptive_obs[:, :, 53] = -proprioceptive_obs[:, :, 56]  # y取负
        flipped_proprioceptive_obs[:, :, 54] =  proprioceptive_obs[:, :, 57]

        # Left hand (55-57) -> Right hand (52-54)
        flipped_proprioceptive_obs[:, :, 55] =  proprioceptive_obs[:, :, 52]  # left_hand
        flipped_proprioceptive_obs[:, :, 56] = -proprioceptive_obs[:, :, 53]  # y取负
        flipped_proprioceptive_obs[:, :, 57] =  proprioceptive_obs[:, :, 54]

        # Right foot (58-60) -> Left foot (61-63)
        flipped_proprioceptive_obs[:, :, 58] =  proprioceptive_obs[:, :, 61]  # right_foot
        flipped_proprioceptive_obs[:, :, 59] = -proprioceptive_obs[:, :, 62]  # y取负
        flipped_proprioceptive_obs[:, :, 60] =  proprioceptive_obs[:, :, 63]

        # Left foot (61-63) -> Right foot (58-60)
        flipped_proprioceptive_obs[:, :, 61] =  proprioceptive_obs[:, :, 58]  # left_foot
        flipped_proprioceptive_obs[:, :, 62] = -proprioceptive_obs[:, :, 59]  # y取负
        flipped_proprioceptive_obs[:, :, 63] =  proprioceptive_obs[:, :, 60]

        return flipped_proprioceptive_obs.view(-1, self.amp_observation_space.shape[0]).detach()

    def flip_roban_critic_obs(self, critic_obs):
        """
        Flip Roban critic observations for symmetry
        Roban critic: ['base_ang_vel', 'projected_gravity', 'velocity_commands', 
        'command_state',     ---add for wbc
        'joint_pos', 'joint_vel', 'actions',
        'base_lin_vel', 'joint_torques', 'joint_accs', 'feet_lin_vel', 'feet_contact_force', 'base_mass_rel', 
        'rigid_body_material', 'base_com', 'action_delay', 'push_force', 'push_torque', 'feet_heights', 'feet_air_times'
        'ref_dof_pos_error', 'height_error', 'bending_error'  ---add for wbc (last two optional; 172-dim critic pads zeros for flip)
        ]
        Roban has 21 joints (with waist_yaw at index 0)
        """
        critic_obs_dim_in = critic_obs.shape[1]
        # 165: same packing as 167 (no command_state, no push) but height_error/bending_error already removed —
        # pad two zeros so downstream sees canonical 167 before cmd/push augmentation.
        strip_wbc_tail = critic_obs_dim_in == 165
        if strip_wbc_tail:
            critic_obs_dim = 167
        else:
            critic_obs_dim = critic_obs_dim_in

        missing_command_state = critic_obs_dim in (173, 203, 167)
        missing_push_terms = critic_obs_dim == 167
        # 172: full Roban critic with cmd+push but without height/bending (174 - 2); pad to 174 for flip.
        missing_wbc_errors = critic_obs_dim == 172
        critic_obs_aug = torch.clone(critic_obs)
        if strip_wbc_tail:
            z_tail = torch.zeros(
                (critic_obs_aug.shape[0], 2),
                device=critic_obs_aug.device,
                dtype=critic_obs_aug.dtype,
            )
            critic_obs_aug = torch.cat([critic_obs_aug, z_tail], dim=1)
        if missing_wbc_errors:
            z_wb = torch.zeros(
                (critic_obs_aug.shape[0], 2),
                device=critic_obs_aug.device,
                dtype=critic_obs_aug.dtype,
            )
            critic_obs_aug = torch.cat([critic_obs_aug, z_wb], dim=1)
        if missing_command_state:
            zeros = torch.zeros((critic_obs_aug.shape[0], 1), device=critic_obs_aug.device, dtype=critic_obs_aug.dtype)
            critic_obs_aug = torch.cat([critic_obs_aug[:, :9], zeros, critic_obs_aug[:, 9:]], dim=1)
        if missing_push_terms:
            # Old fixed mapper expects push_force(3) + push_torque(3) at indices [141:147]
            zeros_push = torch.zeros((critic_obs_aug.shape[0], 6), device=critic_obs_aug.device, dtype=critic_obs_aug.dtype)
            critic_obs_aug = torch.cat([critic_obs_aug[:, :141], zeros_push, critic_obs_aug[:, 141:]], dim=1)
        proprioceptive_obs = critic_obs_aug.view(-1, 1, critic_obs_aug.shape[1])
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        # base_ang_vel (0-2)
        flipped_proprioceptive_obs[:, :, 0] = -proprioceptive_obs[:, :, 0]  # roll
        flipped_proprioceptive_obs[:, :, 1] =  proprioceptive_obs[:, :, 1]  # pitch
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw

        # projected_gravity (3-5)
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]   # x
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]   # y
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]   # z

        # velocity_commands (6-8)
        flipped_proprioceptive_obs[:, :, 6] =  proprioceptive_obs[:, :, 6]   # x command
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 7]   # y command
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]   # yaw command

		# command state for squat and bend 
        flipped_proprioceptive_obs[:, :, 9] = proprioceptive_obs[:, :, 9]    # command state
        
        # joint_pos (10-30): 21 joints
        flipped_proprioceptive_obs[:, :, 10] = -proprioceptive_obs[:, :, 10]   # waist_yaw
        # Left leg (11-16) <- Right leg (17-22)
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]  # left_hip_roll
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]  # left_hip_yaw
        flipped_proprioceptive_obs[:, :, 13] = -proprioceptive_obs[:, :, 19]  # left_hip_pitch
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]  # left_knee
        flipped_proprioceptive_obs[:, :, 15] =  proprioceptive_obs[:, :, 21]  # left_ankle_pitch
        flipped_proprioceptive_obs[:, :, 16] = -proprioceptive_obs[:, :, 22]  # left_ankle_roll

        # Right leg (17-22) <- Left leg (11-16)
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]  # right_hip_roll
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]  # right_hip_yaw
        flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 13]  # right_hip_pitch
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]  # right_knee
        flipped_proprioceptive_obs[:, :, 21] =  proprioceptive_obs[:, :, 15]  # right_ankle_pitch
        flipped_proprioceptive_obs[:, :, 22] = -proprioceptive_obs[:, :, 16]  # right_ankle_roll

        # Left arm (23-26) <- Right arm (27-30)
        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 27]  # left_shoulder_pitch
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 28]  # left_shoulder_roll
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 29]  # left_shoulder_yaw
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 30]  # left_elbow

        # Right arm (27-30) <- Left arm (23-26)
        flipped_proprioceptive_obs[:, :, 27] =  proprioceptive_obs[:, :, 23]  # right_shoulder_pitch
        flipped_proprioceptive_obs[:, :, 28] = -proprioceptive_obs[:, :, 24]  # right_shoulder_roll
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 25]  # right_shoulder_yaw
        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 26]  # right_elbow
      
        # joint_vel (31-51): Same pattern as joint_pos
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 31]  # waist_yaw_vel
        # Left leg vel (32-37) <- Right leg vel (38-43)
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 38]
        flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 39]
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 40]
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 41]
        flipped_proprioceptive_obs[:, :, 36] =  proprioceptive_obs[:, :, 42]
        flipped_proprioceptive_obs[:, :, 37] = -proprioceptive_obs[:, :, 43]

        # Right leg vel (38-43) <- Left leg vel (32-37)
        flipped_proprioceptive_obs[:, :, 38] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 39] = -proprioceptive_obs[:, :, 33]
        flipped_proprioceptive_obs[:, :, 40] = -proprioceptive_obs[:, :, 34]
        flipped_proprioceptive_obs[:, :, 41] =  proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 42] =  proprioceptive_obs[:, :, 36]
        flipped_proprioceptive_obs[:, :, 43] = -proprioceptive_obs[:, :, 37]

        # Left arm vel (44-47) <- Right arm vel (48-51)
        flipped_proprioceptive_obs[:, :, 44] =  proprioceptive_obs[:, :, 48]
        flipped_proprioceptive_obs[:, :, 45] = -proprioceptive_obs[:, :, 49]
        flipped_proprioceptive_obs[:, :, 46] = -proprioceptive_obs[:, :, 50]
        flipped_proprioceptive_obs[:, :, 47] =  proprioceptive_obs[:, :, 51]

        # Right arm vel (48-51) <- Left arm vel (44-47)
        flipped_proprioceptive_obs[:, :, 48] =  proprioceptive_obs[:, :, 44]
        flipped_proprioceptive_obs[:, :, 49] = -proprioceptive_obs[:, :, 45]
        flipped_proprioceptive_obs[:, :, 50] = -proprioceptive_obs[:, :, 46]
        flipped_proprioceptive_obs[:, :, 51] =  proprioceptive_obs[:, :, 47]
        
        # actions (52-72): Same pattern as joint_pos
        flipped_proprioceptive_obs[:, :, 52] = -proprioceptive_obs[:, :, 52]  # waist_yaw_action
        # Left leg actions (53-58) <- Right leg actions (59-64)
        flipped_proprioceptive_obs[:, :, 53] = -proprioceptive_obs[:, :, 59]
        flipped_proprioceptive_obs[:, :, 54] = -proprioceptive_obs[:, :, 60]
        flipped_proprioceptive_obs[:, :, 55] = -proprioceptive_obs[:, :, 61]
        flipped_proprioceptive_obs[:, :, 56] =  proprioceptive_obs[:, :, 62]
        flipped_proprioceptive_obs[:, :, 57] =  proprioceptive_obs[:, :, 63]
        flipped_proprioceptive_obs[:, :, 58] = -proprioceptive_obs[:, :, 64]

        # Right leg actions (59-64) <- Left leg actions (53-58)
        flipped_proprioceptive_obs[:, :, 59] = -proprioceptive_obs[:, :, 53]
        flipped_proprioceptive_obs[:, :, 60] = -proprioceptive_obs[:, :, 54]
        flipped_proprioceptive_obs[:, :, 61] = -proprioceptive_obs[:, :, 55]
        flipped_proprioceptive_obs[:, :, 62] =  proprioceptive_obs[:, :, 56]
        flipped_proprioceptive_obs[:, :, 63] =  proprioceptive_obs[:, :, 57]
        flipped_proprioceptive_obs[:, :, 64] = -proprioceptive_obs[:, :, 58]

        # Left arm actions (65-68) <- Right arm actions (69-72)
        flipped_proprioceptive_obs[:, :, 65] =  proprioceptive_obs[:, :, 69]
        flipped_proprioceptive_obs[:, :, 66] = -proprioceptive_obs[:, :, 70]
        flipped_proprioceptive_obs[:, :, 67] = -proprioceptive_obs[:, :, 71]
        flipped_proprioceptive_obs[:, :, 68] =  proprioceptive_obs[:, :, 72]

        # Right arm actions (69-72) <- Left arm actions (65-68)
        flipped_proprioceptive_obs[:, :, 69] =  proprioceptive_obs[:, :, 65]
        flipped_proprioceptive_obs[:, :, 70] = -proprioceptive_obs[:, :, 66]
        flipped_proprioceptive_obs[:, :, 71] = -proprioceptive_obs[:, :, 67]
        flipped_proprioceptive_obs[:, :, 72] =  proprioceptive_obs[:, :, 68]
        
        # base_lin_vel (73-75)
        flipped_proprioceptive_obs[:, :, 73] =  proprioceptive_obs[:, :, 73]  # x
        flipped_proprioceptive_obs[:, :, 74] = -proprioceptive_obs[:, :, 74]  # y
        flipped_proprioceptive_obs[:, :, 75] =  proprioceptive_obs[:, :, 75]  # z

        # joint_torques (76-96): Same pattern as joint_pos
        flipped_proprioceptive_obs[:, :, 76] = -proprioceptive_obs[:, :, 76]  # waist_yaw_torque
        
        # Left leg torques (77-82) <- Right leg torques (83-88)
        flipped_proprioceptive_obs[:, :, 77] = -proprioceptive_obs[:, :, 83]
        flipped_proprioceptive_obs[:, :, 78] = -proprioceptive_obs[:, :, 84]
        flipped_proprioceptive_obs[:, :, 79] = -proprioceptive_obs[:, :, 85]
        flipped_proprioceptive_obs[:, :, 80] =  proprioceptive_obs[:, :, 86]
        flipped_proprioceptive_obs[:, :, 81] =  proprioceptive_obs[:, :, 87]
        flipped_proprioceptive_obs[:, :, 82] = -proprioceptive_obs[:, :, 88]

        # Right leg torques (83-88) <- Left leg torques (77-82)
        flipped_proprioceptive_obs[:, :, 83] = -proprioceptive_obs[:, :, 77]
        flipped_proprioceptive_obs[:, :, 84] = -proprioceptive_obs[:, :, 78]
        flipped_proprioceptive_obs[:, :, 85] = -proprioceptive_obs[:, :, 79]
        flipped_proprioceptive_obs[:, :, 86] =  proprioceptive_obs[:, :, 80]
        flipped_proprioceptive_obs[:, :, 87] =  proprioceptive_obs[:, :, 81]
        flipped_proprioceptive_obs[:, :, 88] = -proprioceptive_obs[:, :, 82]

        # Left arm torques (89-92) <- Right arm torques (93-96)
        flipped_proprioceptive_obs[:, :, 89] =  proprioceptive_obs[:, :, 93]
        flipped_proprioceptive_obs[:, :, 90] = -proprioceptive_obs[:, :, 94]
        flipped_proprioceptive_obs[:, :, 91] = -proprioceptive_obs[:, :, 95]
        flipped_proprioceptive_obs[:, :, 92] =  proprioceptive_obs[:, :, 96]

        # Right arm torques (93-96) <- Left arm torques (89-92)
        flipped_proprioceptive_obs[:, :, 93] =  proprioceptive_obs[:, :, 89]
        flipped_proprioceptive_obs[:, :, 94] = -proprioceptive_obs[:, :, 90]
        flipped_proprioceptive_obs[:, :, 95] = -proprioceptive_obs[:, :, 91]
        flipped_proprioceptive_obs[:, :, 96] =  proprioceptive_obs[:, :, 92]

        # joint_accs (97-117): Same pattern as joint_pos
        flipped_proprioceptive_obs[:, :, 97] = -proprioceptive_obs[:, :, 97]  # waist_yaw_acc
        # Left leg accs (98-103) <- Right leg accs (104-109)
        flipped_proprioceptive_obs[:, :, 98] = -proprioceptive_obs[:, :, 104]
        flipped_proprioceptive_obs[:, :, 99] = -proprioceptive_obs[:, :, 105]
        flipped_proprioceptive_obs[:, :, 100] = -proprioceptive_obs[:, :, 106]
        flipped_proprioceptive_obs[:, :, 101] =  proprioceptive_obs[:, :, 107]
        flipped_proprioceptive_obs[:, :, 102] =  proprioceptive_obs[:, :, 108]
        flipped_proprioceptive_obs[:, :, 103] = -proprioceptive_obs[:, :, 109]

        # Right leg accs (104-109) <- Left leg accs (98-103)
        flipped_proprioceptive_obs[:, :, 104] = -proprioceptive_obs[:, :, 98]
        flipped_proprioceptive_obs[:, :, 105] = -proprioceptive_obs[:, :, 99]
        flipped_proprioceptive_obs[:, :, 106] = -proprioceptive_obs[:, :, 100]
        flipped_proprioceptive_obs[:, :, 107] =  proprioceptive_obs[:, :, 101]
        flipped_proprioceptive_obs[:, :, 108] =  proprioceptive_obs[:, :, 102]
        flipped_proprioceptive_obs[:, :, 109] = -proprioceptive_obs[:, :, 103]

        # Left arm accs (110-113) <- Right arm accs (114-117)
        flipped_proprioceptive_obs[:, :, 110] =  proprioceptive_obs[:, :, 114]
        flipped_proprioceptive_obs[:, :, 111] = -proprioceptive_obs[:, :, 115]
        flipped_proprioceptive_obs[:, :, 112] = -proprioceptive_obs[:, :, 116]
        flipped_proprioceptive_obs[:, :, 113] =  proprioceptive_obs[:, :, 117]

        # Right arm accs (114-117) <- Left arm accs (110-113)
        flipped_proprioceptive_obs[:, :, 114] =  proprioceptive_obs[:, :, 110]
        flipped_proprioceptive_obs[:, :, 115] = -proprioceptive_obs[:, :, 111]
        flipped_proprioceptive_obs[:, :, 116] = -proprioceptive_obs[:, :, 112]
        flipped_proprioceptive_obs[:, :, 117] =  proprioceptive_obs[:, :, 113]

        # feet_lin_vel (118-123): left <-> right
        flipped_proprioceptive_obs[:, :, 118] =  proprioceptive_obs[:, :, 121]  # left x
        flipped_proprioceptive_obs[:, :, 119] = -proprioceptive_obs[:, :, 122]  # left y
        flipped_proprioceptive_obs[:, :, 120] =  proprioceptive_obs[:, :, 123]  # left z

        flipped_proprioceptive_obs[:, :, 121] =  proprioceptive_obs[:, :, 118]  # right x
        flipped_proprioceptive_obs[:, :, 122] = -proprioceptive_obs[:, :, 119]  # right y
        flipped_proprioceptive_obs[:, :, 123] =  proprioceptive_obs[:, :, 120]  # right z

        # feet_contact_force (124-129): left <-> right
        flipped_proprioceptive_obs[:, :, 124] =  proprioceptive_obs[:, :, 127]  # left x
        flipped_proprioceptive_obs[:, :, 125] = -proprioceptive_obs[:, :, 128]  # left y
        flipped_proprioceptive_obs[:, :, 126] =  proprioceptive_obs[:, :, 129]  # left z

        flipped_proprioceptive_obs[:, :, 127] =  proprioceptive_obs[:, :, 124]  # right x
        flipped_proprioceptive_obs[:, :, 128] = -proprioceptive_obs[:, :, 125]  # right y
        flipped_proprioceptive_obs[:, :, 129] =  proprioceptive_obs[:, :, 126]  # right z

        # base_mass_rel (130)
        flipped_proprioceptive_obs[:, :, 130] =  proprioceptive_obs[:, :, 130]
        # rigid_body_material (131-136): left <-> right
        flipped_proprioceptive_obs[:, :, 131] =  proprioceptive_obs[:, :, 134]
        flipped_proprioceptive_obs[:, :, 132] =  proprioceptive_obs[:, :, 135]
        flipped_proprioceptive_obs[:, :, 133] =  proprioceptive_obs[:, :, 136]
        flipped_proprioceptive_obs[:, :, 134] =  proprioceptive_obs[:, :, 131]
        flipped_proprioceptive_obs[:, :, 135] =  proprioceptive_obs[:, :, 132]
        flipped_proprioceptive_obs[:, :, 136] =  proprioceptive_obs[:, :, 133]

        # base_com (137-139)
        flipped_proprioceptive_obs[:, :, 137] =  proprioceptive_obs[:, :, 137]
        flipped_proprioceptive_obs[:, :, 138] =  proprioceptive_obs[:, :, 138]
        flipped_proprioceptive_obs[:, :, 139] =  proprioceptive_obs[:, :, 139]
        
        # action_delay (140)
        flipped_proprioceptive_obs[:, :, 140] =  proprioceptive_obs[:, :, 140]

        # push_force (141-143)
        flipped_proprioceptive_obs[:, :, 141] =  proprioceptive_obs[:, :, 141]
        flipped_proprioceptive_obs[:, :, 142] = -proprioceptive_obs[:, :, 142]
        flipped_proprioceptive_obs[:, :, 143] =  proprioceptive_obs[:, :, 143]

        # push_torque (144-146)
        flipped_proprioceptive_obs[:, :, 144] = -proprioceptive_obs[:, :, 144]
        flipped_proprioceptive_obs[:, :, 145] =  proprioceptive_obs[:, :, 145]
        flipped_proprioceptive_obs[:, :, 146] = -proprioceptive_obs[:, :, 146]
        
        # feet_heights (147-148): left <-> right
        flipped_proprioceptive_obs[:, :, 147] =  proprioceptive_obs[:, :, 148]
        flipped_proprioceptive_obs[:, :, 148] =  proprioceptive_obs[:, :, 147]

        # feet_air_times (149-150): left <-> right
        flipped_proprioceptive_obs[:, :, 149] =  proprioceptive_obs[:, :, 150]
        flipped_proprioceptive_obs[:, :, 150] =  proprioceptive_obs[:, :, 149]
        
		# 'ref_dof_pos_error' - 只处理腿部（152-163），手臂（164--）不需要对称性

		# joint_pos (151-171): 21 joints
        flipped_proprioceptive_obs[:, :, 151] = -proprioceptive_obs[:, :, 151]   # waist_yaw
		# Left leg (152-157) <- Right leg (158-163)
        flipped_proprioceptive_obs[:, :, 152] = -proprioceptive_obs[:, :, 158]  # left_hip_roll
        flipped_proprioceptive_obs[:, :, 153] = -proprioceptive_obs[:, :, 159]  # left_hip_yaw
        flipped_proprioceptive_obs[:, :, 154] = -proprioceptive_obs[:, :, 160]  # left_hip_pitch
        flipped_proprioceptive_obs[:, :, 155] =  proprioceptive_obs[:, :, 161]  # left_knee
        flipped_proprioceptive_obs[:, :, 156] =  proprioceptive_obs[:, :, 162]  # left_ankle_pitch
        flipped_proprioceptive_obs[:, :, 157] = -proprioceptive_obs[:, :, 163]  # left_ankle_roll        
        # Right leg (158-163) <- Left leg (152-157)
        flipped_proprioceptive_obs[:, :, 158] = -proprioceptive_obs[:, :, 152]  # right_hip_roll
        flipped_proprioceptive_obs[:, :, 159] = -proprioceptive_obs[:, :, 153]  # right_hip_yaw
        flipped_proprioceptive_obs[:, :, 160] = -proprioceptive_obs[:, :, 154]  # right_hip_pitch
        flipped_proprioceptive_obs[:, :, 161] =  proprioceptive_obs[:, :, 155]  # right_knee
        flipped_proprioceptive_obs[:, :, 162] =  proprioceptive_obs[:, :, 156]  # right_ankle_pitch
        flipped_proprioceptive_obs[:, :, 163] = -proprioceptive_obs[:, :, 157]  # right_ankle_roll        
        # 手臂部分（12-25）不做对称变换，直接复制
        flipped_proprioceptive_obs[:, :, 164:172] = -proprioceptive_obs[:, :, 164:172] 
        
        #'height_error', 
        flipped_proprioceptive_obs[:, :, 172] = -proprioceptive_obs[:, :, 172] 
        # 'bending_error'
        flipped_proprioceptive_obs[:, :, 173] = -proprioceptive_obs[:, :, 173] 

        flipped = flipped_proprioceptive_obs.view(-1, critic_obs_aug.shape[1]).detach()
        if missing_push_terms:
            flipped = torch.cat([flipped[:, :141], flipped[:, 147:]], dim=1)
        if missing_command_state:
            flipped = torch.cat([flipped[:, :9], flipped[:, 10:]], dim=1)
        if missing_wbc_errors:
            flipped = flipped[:, :172]
        if strip_wbc_tail:
            flipped = flipped[:, :-2]
        return flipped

    def scale_roban_amp_obs(self, amp_obs):    
        """Scale specific joint observations for Roban (21 joints)
        Scales hip_pitch, knee, ankle_pitch joints (sagittal plane joints)
        Roban has waist_yaw at index 0, so leg indices are shifted by 1
        squat and bend obs for roban: [ "joint_pos", "joint_vel","base_pos_z","projected_gravity", "root lin vel", "root ang vel", "rel key_body pos"
        """
        single_amp_obs = self.amp_observation_space.shape[0]//self._discriminator_history_length
        if amp_obs.dim()==2:    
            amp_obs = amp_obs.view(-1, 1, self.amp_observation_space.shape[0])
        for i in range(self._discriminator_history_length):
            # Left leg joint positions: left_hip_pitch, left_knee, left_ankle_pitch
            amp_obs[:, :, i*single_amp_obs+3] *=2  
            amp_obs[:, :, i*single_amp_obs+4] *=2  
            amp_obs[:, :, i*single_amp_obs+5] *=2  

            # Right leg joint positions: right_hip_pitch, right_knee, right_ankle_pitch
            amp_obs[:, :, i*single_amp_obs+9] *=2  
            amp_obs[:, :, i*single_amp_obs+10] *=2  
            amp_obs[:, :, i*single_amp_obs+11] *=2  

            # Left leg joint velocities (offset by 21 joints)
            amp_obs[:, :, i*single_amp_obs+3+21] *=2  
            amp_obs[:, :, i*single_amp_obs+4+21] *=2  
            amp_obs[:, :, i*single_amp_obs+5+21] *=2  

            # Right leg joint velocities (offset by 21 joints)
            amp_obs[:, :, i*single_amp_obs+9+21] *=2  
            amp_obs[:, :, i*single_amp_obs+10+21] *=2  
            amp_obs[:, :, i*single_amp_obs+11+21] *=2  
        if amp_obs.dim()==2:    
            amp_obs = amp_obs.view(-1,  self.amp_observation_space.shape[0])
        return amp_obs

    def flip_roban_actor_obs(self, obs):
        """
        Flip observation for Roban robot (21 joints).
        # origin Policy obs: ['base_ang_vel', 'projected_gravity', 'velocity_commands',                 'joint_pos', 'joint_vel', 'actions']
        # wbc    policy obs: ['base_ang_vel', 'projected_gravity', 'velocity_commands','command_state', 'joint_pos', 'joint_vel', 'actions']
        Joint order for Roban (21 joints):
        0: waist_yaw_joint
        1-6: leg_l1-l6 (left leg)
        7-12: leg_r1-r6 (right leg)
        13-16: zarm_l1-l4 (left arm, 4 joints)
        17-20: zarm_r1-r4 (right arm, 4 joints)
        
        Total obs: 3 (ang_vel) + 3 (gravity)  +3 (commands) +1 (command state) + 21 (joint_pos) + 21 (joint_vel) + 21 (actions) = 73
        """
        actor_obs_dim = obs.shape[1]
        missing_command_state = actor_obs_dim in (51, 72)
        actor_obs_aug = torch.clone(obs)
        if missing_command_state:
            zeros = torch.zeros((actor_obs_aug.shape[0], 1), device=actor_obs_aug.device, dtype=actor_obs_aug.dtype)
            actor_obs_aug = torch.cat([actor_obs_aug[:, :9], zeros, actor_obs_aug[:, 9:]], dim=1)
        proprioceptive_obs = actor_obs_aug.view(-1, 1, actor_obs_aug.shape[1])
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        # Base angular velocity (0-2)
        flipped_proprioceptive_obs[:, :, 0] = -proprioceptive_obs[:, :, 0]  # roll
        flipped_proprioceptive_obs[:, :, 1] =  proprioceptive_obs[:, :, 1]  # pitch
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # yaw
        
        # Projected gravity (3-5)
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]  # x
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]  # y
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]  # z
        
        # Velocity commands (6-8)
        flipped_proprioceptive_obs[:, :, 6] =  proprioceptive_obs[:, :, 6]  # x
        flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 7]  # y
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]  # yaw

		# command state
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9]  # command state 
        
        # Joint positions (10-30): waist + legs + arms
        # Waist (10)
        flipped_proprioceptive_obs[:, :, 10] = -proprioceptive_obs[:, :, 10]  # waist_yaw (flip sign)
        
        # Left leg -> Right leg (11-16 -> 17-22)
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]  # leg_l1 -> leg_r1
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]  # leg_l2 -> leg_r2
        flipped_proprioceptive_obs[:, :, 13] = -proprioceptive_obs[:, :, 19]  # leg_l3 -> leg_r3
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]  # leg_l4 -> leg_r4
        flipped_proprioceptive_obs[:, :, 15] =  proprioceptive_obs[:, :, 21]  # leg_l5 -> leg_r5
        flipped_proprioceptive_obs[:, :, 16] = -proprioceptive_obs[:, :, 22]  # leg_l6 -> leg_r6
        
        # Right leg -> Left leg (17-22 -> 11-16)
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]  # leg_r1 -> leg_l1
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]  # leg_r2 -> leg_l2
        flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 13]  # leg_r3 -> leg_l3
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]  # leg_r4 -> leg_l4
        flipped_proprioceptive_obs[:, :, 21] =  proprioceptive_obs[:, :, 15]  # leg_r5 -> leg_l5
        flipped_proprioceptive_obs[:, :, 22] = -proprioceptive_obs[:, :, 16]  # leg_r6 -> leg_l6
        
        # Left arm -> Right arm (23-26 -> 27-30)
        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 27]  # zarm_l1 -> zarm_r1
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 28]  # zarm_l2 -> zarm_r2
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 29]  # zarm_l3 -> zarm_r3
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 30]  # zarm_l4 -> zarm_r4
        
        # Right arm -> Left arm (27-30 -> 23-26)
        flipped_proprioceptive_obs[:, :, 27] =  proprioceptive_obs[:, :, 23]  # zarm_r1 -> zarm_l1
        flipped_proprioceptive_obs[:, :, 28] = -proprioceptive_obs[:, :, 24]  # zarm_r2 -> zarm_l2
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 25]  # zarm_r3 -> zarm_l3
        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 26]  # zarm_r4 -> zarm_l4
        
        # Joint velocities (31-51): same pattern as positions
        offset = 22   # add command state --> offset+1=22
        # Waist
        flipped_proprioceptive_obs[:, :, 9+offset] = -proprioceptive_obs[:, :, 9+offset]
        
        # Legs
        flipped_proprioceptive_obs[:, :, 10+offset] = -proprioceptive_obs[:, :, 16+offset]
        flipped_proprioceptive_obs[:, :, 11+offset] = -proprioceptive_obs[:, :, 17+offset]
        flipped_proprioceptive_obs[:, :, 12+offset] = -proprioceptive_obs[:, :, 18+offset]
        flipped_proprioceptive_obs[:, :, 13+offset] =  proprioceptive_obs[:, :, 19+offset]
        flipped_proprioceptive_obs[:, :, 14+offset] =  proprioceptive_obs[:, :, 20+offset]
        flipped_proprioceptive_obs[:, :, 15+offset] = -proprioceptive_obs[:, :, 21+offset]
        
        flipped_proprioceptive_obs[:, :, 16+offset] = -proprioceptive_obs[:, :, 10+offset]
        flipped_proprioceptive_obs[:, :, 17+offset] = -proprioceptive_obs[:, :, 11+offset]
        flipped_proprioceptive_obs[:, :, 18+offset] = -proprioceptive_obs[:, :, 12+offset]
        flipped_proprioceptive_obs[:, :, 19+offset] =  proprioceptive_obs[:, :, 13+offset]
        flipped_proprioceptive_obs[:, :, 20+offset] =  proprioceptive_obs[:, :, 14+offset]
        flipped_proprioceptive_obs[:, :, 21+offset] = -proprioceptive_obs[:, :, 15+offset]
        
        # Arms
        flipped_proprioceptive_obs[:, :, 22+offset] =  proprioceptive_obs[:, :, 26+offset]
        flipped_proprioceptive_obs[:, :, 23+offset] = -proprioceptive_obs[:, :, 27+offset]
        flipped_proprioceptive_obs[:, :, 24+offset] = -proprioceptive_obs[:, :, 28+offset]
        flipped_proprioceptive_obs[:, :, 25+offset] =  proprioceptive_obs[:, :, 29+offset]
        
        flipped_proprioceptive_obs[:, :, 26+offset] =  proprioceptive_obs[:, :, 22+offset]
        flipped_proprioceptive_obs[:, :, 27+offset] = -proprioceptive_obs[:, :, 23+offset]
        flipped_proprioceptive_obs[:, :, 28+offset] = -proprioceptive_obs[:, :, 24+offset]
        flipped_proprioceptive_obs[:, :, 29+offset] =  proprioceptive_obs[:, :, 25+offset]
        
        # Actions (52-72): same pattern as positions
        offset = 43  # add command state --> offset+1= 43
        # Waist
        flipped_proprioceptive_obs[:, :, 9+offset] = -proprioceptive_obs[:, :, 9+offset]
        
        # Legs
        flipped_proprioceptive_obs[:, :, 10+offset] = -proprioceptive_obs[:, :, 16+offset]
        flipped_proprioceptive_obs[:, :, 11+offset] = -proprioceptive_obs[:, :, 17+offset]
        flipped_proprioceptive_obs[:, :, 12+offset] = -proprioceptive_obs[:, :, 18+offset]
        flipped_proprioceptive_obs[:, :, 13+offset] =  proprioceptive_obs[:, :, 19+offset]
        flipped_proprioceptive_obs[:, :, 14+offset] =  proprioceptive_obs[:, :, 20+offset]
        flipped_proprioceptive_obs[:, :, 15+offset] = -proprioceptive_obs[:, :, 21+offset]
        
        flipped_proprioceptive_obs[:, :, 16+offset] = -proprioceptive_obs[:, :, 10+offset]
        flipped_proprioceptive_obs[:, :, 17+offset] = -proprioceptive_obs[:, :, 11+offset]
        flipped_proprioceptive_obs[:, :, 18+offset] = -proprioceptive_obs[:, :, 12+offset]
        flipped_proprioceptive_obs[:, :, 19+offset] =  proprioceptive_obs[:, :, 13+offset]
        flipped_proprioceptive_obs[:, :, 20+offset] =  proprioceptive_obs[:, :, 14+offset]
        flipped_proprioceptive_obs[:, :, 21+offset] = -proprioceptive_obs[:, :, 15+offset]
        
        # Arms
        flipped_proprioceptive_obs[:, :, 22+offset] =  proprioceptive_obs[:, :, 26+offset]
        flipped_proprioceptive_obs[:, :, 23+offset] = -proprioceptive_obs[:, :, 27+offset]
        flipped_proprioceptive_obs[:, :, 24+offset] = -proprioceptive_obs[:, :, 28+offset]
        flipped_proprioceptive_obs[:, :, 25+offset] =  proprioceptive_obs[:, :, 29+offset]
        
        flipped_proprioceptive_obs[:, :, 26+offset] =  proprioceptive_obs[:, :, 22+offset]
        flipped_proprioceptive_obs[:, :, 27+offset] = -proprioceptive_obs[:, :, 23+offset]
        flipped_proprioceptive_obs[:, :, 28+offset] = -proprioceptive_obs[:, :, 24+offset]
        flipped_proprioceptive_obs[:, :, 29+offset] =  proprioceptive_obs[:, :, 25+offset]
        
        flipped = flipped_proprioceptive_obs.view(-1, actor_obs_aug.shape[1]).detach()
        if missing_command_state:
            flipped = torch.cat([flipped[:, :9], flipped[:, 10:]], dim=1)
        return flipped