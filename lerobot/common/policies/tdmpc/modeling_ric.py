import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Tensor, Optional
from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.tdmpc.utils import two_hot_inv, SimNorm
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTModel, VQBeTConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionConfig


@dataclass
class RICConfig:
    multitask: bool = False
    num_tasks: int = 1
    task_dim: int = 32
    mlp_dim: int = 256
    latent_dim: int = 256
    image_encoder_hidden_dim: int = 256
    state_encoder_hidden_dim: int = 256
    num_bins: int = 100
    dropout: float = 0.1

    # Inference.
    use_mpc: bool = True
    cem_iterations: int = 6
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 51
    uncertainty_regularizer_coeff: float = 1.0
    elite_weighting_temperature: float = 0.5
    gaussian_mean_momentum: float = 0.1
    q_ensemble_size: int = 5
    vqbet_config: Optional[VQBeTConfig] = None
    diffusion_config: Optional[DiffusionConfig] = None


class RICModel(nn.Module):
    """Neural network model components for RIC."""
    
    def __init__(self, config: RICConfig):
        super().__init__()
        self.config = config
        
        if self.config.multitask:
            self._task_emb = nn.Embedding(self.config.num_tasks, self.config.task_dim, max_norm=1)
            
        self._encoder = RICObservationEncoder(config)
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
        
        if self.config.vqbet_config is not None:
            self.policy_model = VQBeTModel(self.config.vqbet_config)
        
        if self.config.diffusion_config is not None:
            self.policy_model = DiffusionModel(self.config.diffusion_config)

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
            assert isinstance(m[-1], nn.Linear)
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

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
        """Get task embedding."""
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
        """Enable/disable Q-network gradients."""
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.config.multitask:
            for p in self._task_emb.parameters(): 
                p.requires_grad_(mode)

class RICPolicy(nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "ric"],
):
    """Robust imitation with a critic and implicit world models."""

    def __init__(self, config: RICConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__()
        self.config = config
        self.model = RICModel(config)

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

    def encode(self, obs: dict[str, Tensor], task_index: Tensor) -> Tensor:
        return self.model.encode(obs, task_index)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor, task_index: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.latent_dynamics_and_reward(z, a, task_index)

    def latent_dynamics(self, z: Tensor, a: Tensor, task_index: Tensor) -> Tensor:
        return self.model.latent_dynamics(z, a, task_index)

    def reward(self, z: Tensor, a: Tensor, task_index: Tensor) -> Tensor:
        return self.model.reward(z, a, task_index)

    def Qs(self, z: Tensor, a: Tensor, task_index: Tensor, return_type: str = 'all', target=False) -> Tensor:
        return self.model.Qs(z, a, task_index, return_type, target)


class RICObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: RICConfig):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

        # Identify camera inputs
        self.camera_keys = [key for key in config.input_shapes if "image" in key]
        
        # Shared trunk for all camera inputs
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(config.input_shapes[self.camera_keys[0]][0] + config.task_dim, 
                     config.image_encoder_hidden_dim, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=1),
            nn.ReLU(),
        )

        # Calculate output shape using dummy input
        dummy_batch = torch.zeros(1, *config.input_shapes[self.camera_keys[0]])
        with torch.inference_mode():
            trunk_out_shape = self.shared_trunk(dummy_batch).shape[1:]

        # Shared final layers
        self.shared_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(trunk_out_shape), config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Sigmoid(),
        )

        # Optional camera-specific preprocessing layers if needed
        self.camera_specific = nn.ModuleDict({
            key: nn.Identity() for key in self.camera_keys
        })

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

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        encoded_features = []
        
        # Process image inputs through shared architecture
        for key in self.camera_keys:
            x = self.camera_specific[key](obs[key])
            x = self.shared_trunk(x)
            x = self.shared_final(x)
            encoded_features.append(x)
            
        # Process state inputs
        for input_key, encoder in zip(self.state_encoder_inputs, self.state_encoders):
            if input_key in obs:
                encoded_features.append(encoder(obs[input_key]))
        
        # Combine all features
        if len(encoded_features) > 1:
            return torch.mean(torch.stack(encoded_features), dim=0)
        return encoded_features[0]