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
"""Process pickle files formatted like in the d3il stacking dataset"""

import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    quat2euler,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir):
    state_dir = raw_dir / "state"
    bp_cam_dir = raw_dir / "images" / "bp-cam"
    inhand_cam_dir = raw_dir / "images" / "inhand-cam"

    assert state_dir.exists(), "State directory not found"
    assert bp_cam_dir.exists(), "BP camera images directory not found"
    assert inhand_cam_dir.exists(), "Inhand camera images directory not found"

    state_files = list(state_dir.glob("*.pkl"))
    assert len(state_files) > 0, "No state files found"

    with open(state_files[0], "rb") as f:
        state_data = pickle.load(f)

    assert "robot" in state_data, "Robot data not found in state file"
    assert "des_j_pos" in state_data["robot"], "Joint position data not found"
    assert "gripper_width" in state_data["robot"], "Gripper width data not found"


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    state_dir = raw_dir / "state"
    bp_cam_dir = raw_dir / "images" / "bp-cam"
    inhand_cam_dir = raw_dir / "images" / "inhand-cam"

    state_files = sorted(state_dir.glob("*.pkl"))
    num_episodes = len(state_files)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        state_file = state_files[selected_ep_idx]
        episode_name = state_file.stem

        with open(state_file, "rb") as f:
            state_data = pickle.load(f)

        # extract relevant keys from the state_data dictionary
        robot_des_j_pos = state_data['robot']['des_j_pos']
        robot_j_pos = state_data['robot']['j_pos']
        robot_gripper = np.expand_dims(state_data['robot']['gripper_width'], -1)

        red_box_pos = state_data['red-box']['pos'][:, :2]
        red_box_quat = np.tan(quat2euler(state_data['red-box']['quat']))

        green_box_pos = state_data['green-box']['pos'][:, :2]
        green_box_quat = np.tan(quat2euler(state_data['green-box']['quat']))

        blue_box_pos = state_data['blue-box']['pos'][:, :2]
        blue_box_quat = np.tan(quat2euler(state_data['blue-box']['quat']))

        # Combine the information to create the observation state
        env_state = np.concatenate([
            red_box_pos,
            red_box_quat,
            green_box_pos,
            green_box_quat,
            blue_box_pos,
            blue_box_quat,
        ], axis=-1)
        
        input_state = np.concatenate((robot_j_pos, robot_gripper), axis=-1)
        vel_state = robot_des_j_pos - robot_j_pos
        action = np.concatenate((vel_state, robot_gripper), axis=-1)
        abs_action = np.concatenate((robot_des_j_pos, robot_gripper), axis=-1)

        num_frames = len(input_state)

        ep_dict = {}
        ep_dict["observation.state"] = torch.from_numpy(input_state).float()
        ep_dict["observation.environment_state"] = torch.from_numpy(env_state).float()
        ep_dict["action"] = torch.from_numpy(action).float()
        ep_dict["action_abs"] = torch.from_numpy(abs_action).float()
        # Add next.reward to ep_dict
        action_length = len(action)
        next_reward = torch.zeros(action_length)
        next_reward[-1] = 1.0  # Set the last reward to 1
        ep_dict["next.reward"] = next_reward
        ep_dict["next.done"] = torch.zeros(action_length, dtype=torch.bool)
        ep_dict["next.done"][-1] = True
        ep_dict["next.success"] = torch.zeros(action_length, dtype=torch.bool)
        ep_dict["next.success"][-1] = True

        # Load and process BP camera images
        bp_imgs = sorted((bp_cam_dir / episode_name).glob("*.jpg"), key=lambda x: int(x.stem))
        bp_img_key = "observation.images.bp_cam"
        if video:
            bp_video_path = process_images_to_video(bp_imgs, videos_dir, bp_img_key, ep_idx, fps, encoding)
            ep_dict[bp_img_key] = [{"path": f"videos/{bp_video_path.name}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[bp_img_key] = [PILImage.open(img_path) for img_path in bp_imgs[:num_frames]]

        # Load and process inhand camera images
        inhand_imgs = sorted((inhand_cam_dir / episode_name).glob("*.jpg"), key=lambda x: int(x.stem))
        inhand_img_key = "observation.images.inhand_cam"
        if video:
            inhand_video_path = process_images_to_video(inhand_imgs, videos_dir, inhand_img_key, ep_idx, fps, encoding)
            ep_dict[inhand_img_key] = [{"path": f"videos/{inhand_video_path.name}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[inhand_img_key] = [PILImage.open(img_path) for img_path in inhand_imgs[:num_frames]]

        ep_dict["episode_index"] = torch.full((num_frames,), ep_idx, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def process_images_to_video(img_paths, videos_dir, key, ep_idx, fps, encoding):
    tmp_imgs_dir = videos_dir / "tmp_images"
    tmp_imgs_dir.mkdir(parents=True, exist_ok=True)

    # Determine the file extension of the first image
    img_format = img_paths[0].suffix[1:]  # Remove the leading dot

    for i, img_path in enumerate(img_paths):
        shutil.copy(img_path, tmp_imgs_dir / f"{i:06d}.{img_format}")

    video_path = videos_dir / f"{key}_episode_{ep_idx:06d}.mp4"
    encode_video_frames(tmp_imgs_dir, video_path, fps, crf=None, img_format=img_format, **(encoding or {}))

    shutil.rmtree(tmp_imgs_dir)
    return video_path


def to_hf_dataset(data_dict, video):
    features = {
        "observation.state": Sequence(feature=Value(dtype="float32", id=None)),
        "observation.environment_state": Sequence(feature=Value(dtype="float32", id=None)),
        "action": Sequence(feature=Value(dtype="float32", id=None)),
        "action_abs": Sequence(feature=Value(dtype="float32", id=None)),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "index": Value(dtype="int64", id=None),
        "next.reward": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        "next.success": Value(dtype="bool", id=None),
    }

    if video:
        features["observation.images.bp_cam"] = VideoFrame()
        features["observation.images.inhand_cam"] = VideoFrame()
    else:
        features["observation.images.bp_cam"] = Image()
        features["observation.images.inhand_cam"] = Image()

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    check_format(raw_dir)

    if fps is None:
        fps = 1 / 0.03  # Assuming 30 FPS, adjust if needed

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    import os
    d3il_dir = Path(os.environ.get("D3IL_SIM_DATASET_DIR", "/home/ksrini/Temporary_D3IL/"))
    raw_dir = d3il_dir / "environments/dataset/data/stacking/vision_data"
    videos_dir =  Path("data") / "demos"
    video = True  
    episodes = [0, 1, 2]
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(raw_dir, videos_dir, video=video, episodes=episodes)
    print(hf_dataset)
    print(episode_data_index)