import pickle
from pathlib import Path

import numpy as np
import torch
import shutil
import tqdm
from datasets import Dataset, Features, Sequence, Value, Image

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from PIL import Image as PILImage

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
    videos_dir: Path = None,
    fps: int = 30,
    video: bool = False,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    bp_cam_dir = raw_dir / "images" / "bp-cam"
    inhand_cam_dir = raw_dir / "images" / "inhand-cam"
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
        next_reward = torch.zeros(num_frames)
        next_reward[-1] = 1.0  # Set the last reward to 1
        ep_dict["next.reward"] = next_reward
        ep_dict["next.done"] = torch.zeros(num_frames, dtype=torch.bool)
        ep_dict["next.done"][-1] = True
        ep_dict["next.success"] = torch.zeros(num_frames, dtype=torch.bool)
        ep_dict["next.success"][-1] = True


        ep_dict["episode_index"] = torch.full((num_frames,), ep_idx, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / 30  # Assuming 30 FPS, adjust if needed

        # Assume `bp_cam_dir` and `inhand_cam_dir` are provided or defined similarly.
        bp_imgs = sorted((bp_cam_dir / episode_name).glob("*.jpg"), key=lambda x: int(x.stem))
        if video:   
            bp_video_path = process_images_to_video(bp_imgs, videos_dir, "observation.images.bp_cam", ep_idx, fps, encoding)
            ep_dict["observation.images.bp_cam"] = [{"path": f"videos/{bp_video_path.name}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict["observation.images.bp_cam"] = [PILImage.open(img_path) for img_path in bp_imgs[:num_frames]]

        inhand_imgs = sorted((inhand_cam_dir / episode_name).glob("*.jpg"), key=lambda x: int(x.stem))
        if video:
            inhand_video_path = process_images_to_video(inhand_imgs, videos_dir, "observation.images.inhand_cam", ep_idx, fps, encoding)
            ep_dict["observation.images.inhand_cam"] = [{"path": f"videos/{inhand_video_path.name}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict["observation.images.inhand_cam"] = [PILImage.open(img_path) for img_path in inhand_imgs[:num_frames]]


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
        "action": Sequence(feature=Value(dtype="float32", id=None)),
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
    videos_dir: Path = None,
    fps: int = 30,  # Assuming default 30 FPS
    video: bool = False,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    check_format(raw_dir)

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }

    return hf_dataset, episode_data_index, info
