#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of TD-MPC2.

The comments in this code may sometimes refer to these references:
    TD-MPC paper: Temporal Difference Learning for Model Predictive Control (https://arxiv.org/abs/2203.04955)
    TD-MPC2 paper: TD-MPC2: Scalable, Robust World Models for Continuous Control (https://www.tdmpc2.com/)

"""

# ruff: noqa: N806

import logging
from collections import deque
from functools import partial
from typing import Callable
from copy import deepcopy

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPC2Config
from lerobot.common.policies.tdmpc.modeling_tdmpc import random_shifts_aug, flatten_forward_unflatten
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues


@torch.jit.script
def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
	return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
	return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
	"""Compute Gaussian log probability."""
	residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
	if size is None:
		size = eps.size(-1)
	return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
	return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	log_pi -= _squash(pi).sum(-1, keepdim=True)
	return mu, pi, log_pi


@torch.jit.script
def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets."""
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
	soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


DREG_BINS = None


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	global DREG_BINS
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	if DREG_BINS is None:
		DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
	return symexp(x)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class TDMPC2Policy(nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "tdmpc2"],
):
    """Implementation of TD-MPC2 learning + inference.
    """

    name = "tdmpc2"

    def __init__(
        self, config: TDMPC2Config | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()

        if config is None:
            config = TDMPC2Config()
        self.config = config
        self.model = TDMPC2WorldModel(config)

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
    )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: This check is covered in the post-init of the config but have a sanity check just in case.
        # assert len(image_keys) == 1
        if len(image_keys) > 0:
            assert len(image_keys) == 1
            self.input_image_key = image_keys[0]
            self._use_image = True
        else:
            self._use_image = False
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True
        if "observation.state" in config.input_shapes:
            self._use_agent_pos = True

        self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in config.episode_lengths], device='cuda'
		) if config.multitask else torch.tensor(self._get_discount(config.episode_length), device='cuda')

        self.reset()


    def _get_discount(self, episode_length): 
        """
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		""" 
        frac = episode_length/self.config.discount_denom
        return min(max((frac-1)/(frac), self.config.discount_min), self.config.discount_max)

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            "action": deque(maxlen=max(self.config.n_action_repeats, self.config.n_action_steps)),
        }
        if self._use_agent_pos:
            self._queues["observation.state"] = deque(maxlen=1)
        if self._use_image:
            self._queues["observation.image"] = deque(maxlen=1)
        if self._use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=1)
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch["observation.image"] = batch[self.input_image_key]

        self._queues = populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            # Remove the time dimensions as it is not handled yet.
            for key in batch:
                assert batch[key].shape[1] == 1
                batch[key] = batch[key][:, 0]

            # NOTE: Order of observations matters here. 
            encode_keys = []
            if self._use_agent_pos:
                encode_keys.append("observation.state")
            if self._use_image:
                encode_keys.append("observation.image")
            if self._use_env_state:
                encode_keys.append("observation.environment_state")
            task_index = batch.get("task_index", torch.ones(batch[encode_keys[0]].shape[0]))
            z = self.model.encode({k: batch[k] for k in encode_keys}, task_index)
            if self.config.use_mpc:
                action = self.plan(z, task_index)
            else:
                action = self.model.pi(z, task_index)[1]

            action = self.unnormalize_outputs({"action": action.clamp(-1, 1)})["action"]

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues["action"].append(action[0])
            else:
                self._queues["action"].extend(action[: self.config.n_action_steps])

        action = self._queues["action"].popleft()
        return action

    @torch.no_grad()
    def plan(self, z: Tensor, task_index: Tensor) -> Tensor:
        """Plan next action using TD-MPC inference.

        Args:
            z: (latent_dim,) tensor for the initial state.
        Returns:
            (action_dim,) tensor for the next action.

        TODO(alexander-soare) Extend this to be able to work with batches.
        """
        device = get_device_from_parameters(self)
        batch_size = z.shape[0]

        # Sample Nπ trajectories from the policy.
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.output_shapes["action"][0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            _task_index = einops.repeat(task_index, "b -> n b", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi(_z, _task_index)[1]
                _z = self.model.latent_dynamics(_z, pi_actions[t], _task_index)

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)
        task_index = einops.repeat(task_index, "b -> n b", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = torch.zeros(self.config.horizon, batch_size, self.config.output_shapes["action"][0], device=device)
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.output_shapes["action"][0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # Compute elite actions.
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions, task_index).nan_to_num_(0)  # (n_gaussian_samples + n_pi_samples, batch)
            elite_idxs = torch.topk(value, self.config.n_elites, dim=0).indices  # (n_elites, batch)
            expanded_elite_idxs = elite_idxs.unsqueeze(0).unsqueeze(-1).expand(self.config.horizon, -1, -1, actions.shape[-1])

            elite_value, elite_actions = torch.gather(value, 0, elite_idxs), torch.gather(actions, 1, expanded_elite_idxs)

            # Update guassian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(dim=0, keepdim=True)[0]
            # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
            # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
            # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(dim=0, keepdim=True)
            score = score.unsqueeze(0).unsqueeze(-1)
            _mean = torch.sum(score * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(score * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1)
            )
            # Update mean with an exponential moving average, and std with a direct replacement.
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp_(self.config.min_std, self.config.max_std)

        # Keep track of the mean for warm-starting subsequent steps.
        self._prev_mean = mean

        # Randomly select one of the elite actions from the last iteration of MPPI/CEM using the softmax
        # scores from the last iteration. 
        sampled_action_indices = torch.multinomial(score.squeeze().T, 1).T.unsqueeze(0).unsqueeze(-1).expand(self.config.horizon, -1, -1, self.config.output_shapes['action'][0])
        actions = elite_actions.gather(1, sampled_action_indices).squeeze(1)

        return actions

    @torch.no_grad()
    def estimate_value(self, z: Tensor, actions: Tensor, task_index: Tensor):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.config.horizon):
            reward = two_hot_inv(self.model.reward(z, actions[t], task_index), self.config).squeeze(-1)
            z = self.model.latent_dynamics(z, actions[t], task_index)
            G += discount * reward
            discount *= self.config.discount
        G = G + discount * self.model.Qs(z, self.model.pi(z, task_index)[1], task_index, return_type='avg').squeeze(-1)
        return G

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.config.multitask else self.discount
        qs = self.model.Qs(next_z, pi, task, return_type='min', target=True)
        return reward + discount * qs.squeeze()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss."""
        device = get_device_from_parameters(self)

        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch["observation.image"] = batch[self.input_image_key]
        batch = self.normalize_targets(batch)

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]  # (t, b)
        reward = batch["next.reward"]  # (t,)
        task_index = batch.get("task_index", torch.ones(batch["action"].shape[1], device=device, dtype=torch.long))
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Apply random image augmentationse
        if self._use_image and self.config.max_random_shift_ratio > 0:
            observations["observation.image"] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations["observation.image"],
            )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon = self.config.horizon


        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        batch_size = batch["index"].shape[0]
        zs = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        zs[0] = self.model.encode(current_observation, task_index)
        reward_preds = torch.empty(horizon, batch_size, self.config.num_bins, device=device)
        for t in range(horizon):
            zs[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(zs[t], action[t], task_index)

        # Compute Q and V value predictions based on the latent rollout.
        _zs = zs[:-1]

        # Compute Q and V value predictions based on the latent rollout.
        q_preds_ensemble = self.model.Qs(_zs, action, task_index, return_type="all")  # (ensemble, horizon, batch)
        info.update({"Q": torch.cat([two_hot_inv(q, self.config) for q in q_preds_ensemble], dim=0).mean(dim=0).mean().item()})

        # Compute various targets with stopgrad.
        with torch.no_grad():
            next_z = self.model.encode(next_observations, task_index)
            td_targets = self._td_target(next_z, reward, task_index)

        # Compute losses.
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        temporal_loss_coeffs = torch.pow(
            self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        ).unsqueeze(-1)
        # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # predicted from the (target model's) observation encoder.
        consistency_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(zs[1:], next_z, reduction="none").mean(dim=-1)
                # `z_preds` depends on the current observation and the actions.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # `z_targets` depends on the next observation.
                * ~batch["observation.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        reward_loss = 0
        for t in range(horizon):
            reward_loss += (
                soft_ce(reward_preds[t], reward[t], self.config)
                * ~batch["next.reward_is_pad"][t]
                * ~batch["observation.state_is_pad"][t]
                * ~batch["action_is_pad"][t]
            ).mean() * self.config.temporal_decay_coeff**t
        
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        value_loss = 0
        for q in range(self.config.q_ensemble_size):
            for t in range(horizon):
                value_loss += (
                    soft_ce(q_preds_ensemble[q][t], td_targets[t], self.config)
                    * ~batch["next.reward_is_pad"][t]
                    * ~batch["observation.state_is_pad"][t]
                    * ~batch["action_is_pad"][t]
                ).mean() * self.config.temporal_decay_coeff**t

        # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # We won't need these gradients again so detach.
        z_preds = zs.detach()

        _, action_preds, log_pis, _ = self.model.pi(z_preds, task_index)  # (t, b, a)
        log_pis = log_pis.squeeze()
        qs = self.model.Qs(z_preds, action_preds, task_index, return_type='avg').squeeze()  # (t, b)
        # q_preds = self.model.Qs(z_preds[:-1], action_preds, task_index, return_type='avg').squeeze()  # (t, b)
        # Calculate the MSE between the /lossactions and the action predictions.
        # Note: FOWM's original code calculates the log probability (wrt to a unit standard deviation
        # gaussian) and sums over the action dimension. Computing the log probability amounts to multiplying
        # the MSE by 0.5 and adding a constant offset (the log(2*pi) term) . Here we drop the constant offset
        # as it doesn't change the optimization step, and we drop the 0.5 as we instead make a configuration
        # parameter for it (see below where we compute the total loss).

        if self.config.pi_loss == "entropy":
            pi_loss = (self.config.entropy_coef * log_pis - qs)
            mask = ~batch["observation.state_is_pad"]
        else:
            pi_loss = F.mse_loss(action_preds, action, reduction="none").sum(-1)
            mask = ~batch["action_is_pad"] * ~batch["observation.state_is_pad"][0]

        rho = torch.pow(self.config.temporal_decay_coeff, torch.arange(len(pi_loss), device=device)).unsqueeze(-1)

        # NOTE: The original implementation does not take the sum over the temporal dimension like with the
        # other losses.
        # TODO(alexander-soare): Take the sum over the temporal dimension and check that training still works
        # as well as expected.

        pi_loss = self.config.pi_coeff * (pi_loss * rho * mask).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * value_loss
            # + self.config.pi_coeff * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": value_loss.item(),
                "pi_loss": pi_loss,
                # "pi_loss": pi_loss.item(),
                "loss": loss,
                "sum_loss": loss.item() * self.config.horizon,
            }
        )

        # Undo (b, t) -> (t, b).
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return info

class TDMPC2WorldModel(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC2."""

    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config
        if self.config.multitask:
            self._task_emb = nn.Embedding(self.config.num_tasks, self.config.task_dim, max_norm=1)
        self._encoder = TDMPCObservationEncoder(config)
        self._dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.output_shapes["action"][0] + config.task_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            SimNorm(config),
        )
        self._reward = nn.Sequential(
            nn.Linear(config.latent_dim + config.output_shapes["action"][0] + config.task_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, max(config.num_bins, 1)),
        )
        self._pi = nn.Sequential(
            nn.Linear(config.latent_dim + config.task_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, 2 * config.output_shapes["action"][0]),
        )
        self.register_buffer("log_std_min", torch.tensor(config.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(config.log_std_max - config.log_std_min))

        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent_dim + config.output_shapes["action"][0] + config.task_dim, config.mlp_dim),
                    nn.Dropout(config.dropout, inplace=True),
                    nn.LayerNorm(config.mlp_dim),
                    nn.Tanh(),
                    nn.Linear(config.mlp_dim, config.mlp_dim),
                    nn.ELU(),
                    nn.Linear(config.mlp_dim, max(config.num_bins, 1)),
                )
                for _ in range(config.q_ensemble_size)
            ]
        )
        self._init_weights()
        self._reward[-1].weight.data.fill_(0)
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

    def _init_weights(self):
        """Initialize model weights.

        Orthogonal initialization for all linear and convolutional layers' weights (apart from final layers
        of reward network and Q networks which get zero initialization).
        Zero initialization for all linear and convolutional layers' biases.
        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight.data, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_apply_fn)
        for m in [self._reward, *self._Qs]:
            assert isinstance(
                m[-1], nn.Linear
            ), "Sanity check. The last linear layer needs 0 initialization on weights."
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)  # this has already been done, but keep this line here for good measure

    def encode(self, obs: dict[str, Tensor], task_index: Tensor) -> Tensor:
        """Encodes an observation into its latent representation."""
        if self.config.multitask:
            obs = self.task_emb(obs, task_index)
        encoded_feat = self._encoder(obs)
        return encoded_feat

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor, task_index: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the next state's latent representation and the reward given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            A tuple containing:
                - (*, latent_dim) tensor for the next state's latent representation.
                - (*,) tensor for the estimated reward.
        """
        return self.latent_dynamics(z, a, task_index), self.reward(z, a, task_index).squeeze(-1)

    def latent_dynamics(self, z: Tensor, a: Tensor, task_index: Tensor) -> Tensor:
        """Predict the next state's latent representation given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            (*, latent_dim) tensor for the next state's latent representation.
        """
        if self.config.multitask:
            z = self.task_emb(z, task_index)
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def reward(self, z: Tensor, a: Tensor, task_index: Tensor) -> Tensor:
        """Predict the reward given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            task_index: (*,) tensor for the task index.
        Returns:
            (*,) tensor for the estimated reward.
        """
        if self.config.multitask:
            z = self.task_emb(z, task_index)
        x = torch.cat([z, a], dim=-1)
        return self._reward(x).squeeze(-1)

    def pi(self, z: Tensor, task_index: Tensor) -> Tensor:
        """Samples an action from the learned policy.

        The policy can also have added (truncated) Gaussian noise injected for encouraging exploration when
        generating rollouts for online training.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            task_index: (*,) tensor for the task index.
        Returns:
            (*, action_dim) tensor for the sampled action.
        """
        if self.config.multitask:
            z = self.task_emb(z, task_index)
        mu, log_std_x = self._pi(z).chunk(2, dim=-1)
        log_std_x = log_std(log_std_x, self.log_std_min, self.log_std_dif) 
        eps = torch.randn_like(mu) 
        log_pi = gaussian_logprob(eps, log_std_x, size=self.config.output_shapes["action"][0]) 
        action = mu + eps * log_std_x.exp() 
        mu, action, log_pi = squash(mu, action, log_pi) 
        return mu, action, log_pi, log_std_x

    def Qs(self, z: Tensor, a: Tensor, task_index: Tensor, return_type: str = 'all', target=False) -> Tensor:  # noqa: N802
        """Predict state-action value for all of the learned Q functions.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            task_index: (*,) tensor for the task index.
            return_type: can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
        Returns:
            (q_ensemble, *) tensor for the value predictions of each learned Q function in the ensemble OR
            (*,) tensor if return_min=True.
        """
        assert return_type in {'min', 'avg', 'all'}
        if self.config.multitask:
            z = self.task_emb(z, task_index)
        x = torch.cat([z, a], dim=-1) 
        qs = self._Qs if not target else self._target_Qs
        out = torch.stack([q(x).squeeze(-1) for q in qs], dim=0)
        if return_type == 'all':
            return out
        else:
            if self.config.q_ensemble_size > 2:  # noqa: SIM108
                out = [out[i] for i in np.random.choice(len(self._Qs), size=2)]
            q1, q2 = two_hot_inv(out[0], self.config), two_hot_inv(out[1], self.config)
            if return_type == 'min':
                return torch.min(q1, q2)
            elif return_type == 'avg':
                return (q1 + q2) / 2
            # else:
            #     return torch.stack([two_hot_inv(q(x).squeeze(-1), self.config) for q in qs], dim=0).mean(dim=0)

    def task_emb(self, z: Tensor | dict[str, Tensor], task_index: Tensor) -> Tensor:
        if isinstance(task_index, int):
            task_index = torch.tensor([task_index], device=z.device) 
        if isinstance(z, dict): 
            for k in z:
                z[k] = self.task_emb(z[k], task_index)
            return z
        else:
            emb = self._task_emb(task_index.long())
            if z.ndim == 5:
                emb = emb.view(1, emb.shape[0], 1, emb.shape[1], 1).repeat(z.shape[0], 1, 1, 1, z.shape[-1])
                return torch.cat([z, emb], dim=2) 
            elif z.ndim == 4:
                emb = emb.view(emb.shape[0], 1, emb.shape[1], 1).repeat(1, 1, 1, z.shape[-1])
                return torch.cat([z, emb], dim=1)
            elif z.ndim == 3:
                emb = emb.unsqueeze(0).repeat(z.shape[0], 1, 1)
            elif emb.shape[0] == 1:
                emb = emb.repeat(z.shape[0], 1)
            return torch.cat([z, emb], dim=-1) 
        
    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.config.multitask:
            for p in self._task_emb.parameters(): 
                p.requires_grad_(mode)


class TDMPCObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPC2Config):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

        if "observation.image" in config.input_shapes:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    config.input_shapes["observation.image"][0] + config.task_dim, config.image_encoder_hidden_dim, 7, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=1),
                nn.ReLU(),
            )
            dummy_batch = torch.zeros(1, *config.input_shapes["observation.image"])
            with torch.inference_mode():
                out_shape = self.image_enc_layers(dummy_batch).shape[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Sigmoid(),
                )
            )
        self.state_encoder_inputs = [ ]
        self.state_encoders = nn.ModuleList()
        for input_shape_key in config.input_shapes:
            input_shape = config.input_shapes[input_shape_key]
            if len(input_shape) == 1:
                state_enc_layers = nn.Sequential(
                     nn.Linear(input_shape[0] + config.task_dim, config.state_encoder_hidden_dim),
                     nn.ELU(),
                     nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                     nn.LayerNorm(config.latent_dim),
                     nn.Sigmoid(),
                )
                self.state_encoder_inputs.append(input_shape_key)
                self.state_encoders.append(state_enc_layers)

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        if "observation.image" in self.config.input_shapes:
            feat.append(flatten_forward_unflatten(self.image_enc_layers, obs_dict["observation.image"]))
        for k, enc in zip(self.state_encoder_inputs, self.state_encoders):
            feat.append(enc(obs_dict[k]))
        feat = torch.stack(feat, dim=0).mean(0)
        return feat


class TDMPC2Optimizer(torch.optim.Adam):
    def __init__(self, policy, cfg):
        self.model_params = [
            {'params': policy.model._encoder.parameters(), 'lr': cfg.training.lr * cfg.policy.enc_lr_scale},
            {'params': policy.model._dynamics.parameters()},
            {'params': policy.model._reward.parameters()},
            {'params': policy.model._Qs.parameters()},
            {'params': policy.model._task_emb.parameters() if cfg.policy.multitask else []}
        ]
        self.pi_params = [{'params': policy.model._pi.parameters()}]

        all_params = self.model_params + self.pi_params
        super().__init__(all_params, lr=cfg.training.lr)

