import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Sequence, Value

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)

def check_format(raw_dir):
    state_files = list(raw_dir.glob("*.pkl"))
    assert len(state_files) > 0, "No state files found"

    with open(state_files[0], "rb") as f:
        state_data = pickle.load(f)

    assert "robot" in state_data, "Robot data not found in state file"
    assert "des_c_pos" in state_data["robot"], "Robot desired position data not found"
    assert "c_pos" in state_data["robot"], "Robot current position data not found"

def load_from_raw(
    raw_dir: Path,
    episodes: list[int] | None = None,
):
    state_files = sorted(raw_dir.glob("*.pkl"))
    num_episodes = len(state_files)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)

    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        state_file = state_files[selected_ep_idx]
        episode_name = state_file.stem

        with open(state_file, "rb") as f:
            env_state = pickle.load(f)

        robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
        robot_c_pos = env_state['robot']['c_pos'][:, :2]

        input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)
        vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
        num_frames = len(vel_state)

        ep_dict = {}
        ep_dict["observation.state"] = torch.from_numpy(input_state[:-1]).float()
        ep_dict["action"] = torch.from_numpy(vel_state).float()

        ep_dict["episode_index"] = torch.full((num_frames,), ep_idx, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / 30  # Assuming 30 FPS, adjust if needed

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict

def to_hf_dataset(data_dict):
    features = {
        "observation.state": Sequence(feature=Value(dtype="float32", id=None)),
        "action": Sequence(feature=Value(dtype="float32", id=None)),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "index": Value(dtype="int64", id=None),
    }

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    episodes: list[int] | None = None,
):
    check_format(raw_dir)

    data_dict = load_from_raw(raw_dir, episodes)
    hf_dataset = to_hf_dataset(data_dict)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": 30,  # Assuming 30 FPS, adjust if needed
    }

    return hf_dataset, episode_data_index, info