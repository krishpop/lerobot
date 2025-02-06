#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import inspect
import logging
import os

from omegaconf import DictConfig, OmegaConf

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.policies.tdmpc.modeling_tdmpc2 import TDMPC2Policy

def _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg):
    expected_kwargs = set(inspect.signature(policy_cfg_class).parameters)
    if not set(hydra_cfg.policy).issuperset(expected_kwargs):
        logging.warning(
            f"Hydra config is missing arguments: {set(expected_kwargs).difference(hydra_cfg.policy)}"
        )

    # OmegaConf.to_container returns lists where sequences are found, but our dataclasses use tuples to avoid
    # issues with mutable defaults. This filter changes all lists to tuples.
    def list_to_tuple(item):
        return tuple(item) if isinstance(item, list) else item

    policy_cfg = policy_cfg_class(
        **{
            k: list_to_tuple(v)
            for k, v in OmegaConf.to_container(hydra_cfg.policy, resolve=True).items()
            if k in expected_kwargs
        }
    )
    return policy_cfg


def get_policy_and_config_classes(name: str) -> tuple[Policy, object]:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy, TDMPCConfig
    if name == "tdmpc2":
        from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPC2Config
        from lerobot.common.policies.tdmpc.modeling_tdmpc2 import TDMPC2Policy

        return TDMPC2Policy, TDMPC2Config
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy, DiffusionConfig
    elif name == "act":
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy, ACTConfig
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy, VQBeTConfig
    elif name == "ric":
        from lerobot.common.policies.tdmpc.modeling_ric import RICModel, RICConfig
        return RICModel, RICConfig
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_critic(hydra_cfg: DictConfig, policy: Policy):
    from pathlib import Path

    from lerobot.common.logger import Logger
    from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCCritic
    from lerobot.common.utils.utils import init_hydra_config

    pretrained_critic_path = hydra_cfg.critic_pretrained_policy_path
    assert Path(pretrained_critic_path).exists(), f"Pretrained critic path {pretrained_critic_path} does not exist"

    if ".pt" in pretrained_critic_path:
        from tdmpc2 import TDMPC2
        from common.parser import parse_cfg
        from envs import make_env
        # trained with tdmpc2 repo
        tdmpc2_config_path = "/juno/u/bsud2/multi_task_experts/tdmpc2/tdmpc2/config.yaml"
        work_dir = "/".join(pretrained_critic_path.split("/")[:-1])
        task = "d3il-stacking" if "stacking" in pretrained_critic_path else "d3il-sorting"
        print("task ", task)
        print("work dir ", work_dir)
        print("pretrained critic path ", pretrained_critic_path)
        overrides = [
            f"task={task}",
            "model_size=48",
            f"work_dir={work_dir}",
            "horizon=6",
            "mpc=false",
            f"checkpoint={pretrained_critic_path}"
        ]
        tdmpc2_cfg = init_hydra_config(tdmpc2_config_path, overrides)
        tdmpc2_cfg = parse_cfg(tdmpc2_cfg)
        make_env(tdmpc2_cfg)
        agent = TDMPC2(tdmpc2_cfg)
        assert os.path.exists(tdmpc2_cfg.checkpoint), f'Checkpoint {tdmpc2_cfg.checkpoint} not found! Must be a valid filepath.'
        agent.load(tdmpc2_cfg.checkpoint)
        return agent
    else:
        last_pretrained_model_dir = Logger.get_last_pretrained_model_dir(pretrained_critic_path)
        assert last_pretrained_model_dir.exists(), f"Last pretrained model dir {last_pretrained_model_dir} does not exist"

        if last_pretrained_model_dir is not None and last_pretrained_model_dir.exists():
            critic_cfg = init_hydra_config(str(last_pretrained_model_dir / "config.yaml"))

        critic_policy = make_policy(critic_cfg, last_pretrained_model_dir, dataset_stats=None)
        if isinstance(critic_policy, TDMPC2Policy):
            critic = TDMPCCritic(critic_policy, use_advantage=False)
        else:
            critic = TDMPCCritic(critic_policy, use_advantage=hydra_cfg.distillation.critic_use_advantage)
        critic.set_normalize_stats(policy)
        return critic


def make_policy(
    hydra_cfg: DictConfig, pretrained_policy_name_or_path: str | None = None, dataset_stats=None
) -> Policy:
    """Make an instance of a policy class."""
    policy_cls, policy_cfg_class = get_policy_and_config_classes(hydra_cfg.policy.name)

    # Handle nested configs for RIC
    if hydra_cfg.policy.name == "ric":
        # Get VQBeT config if specified
        vqbet_config = None
        if "vqbet_config" in hydra_cfg:
            _, vqbet_cfg_class = get_policy_and_config_classes("vqbet")
            vqbet_config = _policy_cfg_from_hydra_cfg(vqbet_cfg_class, hydra_cfg.vqbet_config)
            
        # Get Diffusion config if specified  
        diffusion_config = None
        if "diffusion_config" in hydra_cfg:
            _, diffusion_cfg_class = get_policy_and_config_classes("diffusion")
            diffusion_config = _policy_cfg_from_hydra_cfg(diffusion_cfg_class, hydra_cfg.diffusion_config)

        # Create RIC config with nested configs
        policy_cfg = policy_cfg_class(
            vqbet_config=vqbet_config,
            diffusion_config=diffusion_config,
            **{k: v for k, v in OmegaConf.to_container(hydra_cfg.policy).items() 
               if k not in ["vqbet_config", "diffusion_config"]}
        )
    else:
        policy_cfg = _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg)

    if pretrained_policy_name_or_path is None:
        # Make a fresh policy
        policy = policy_cls(policy_cfg, dataset_stats)
    else:
        # Load pretrained policy
        policy = policy_cls(policy_cfg)
        policy.load_state_dict(policy_cls.from_pretrained(pretrained_policy_name_or_path).state_dict())

    policy.to(get_safe_torch_device(hydra_cfg.device))

    return policy
