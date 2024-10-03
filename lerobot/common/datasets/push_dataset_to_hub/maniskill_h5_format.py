import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

import shutil

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

def check_format(h5_file):
    with h5py.File(h5_file, 'r') as f:
        assert 'traj_0' in f, "No trajectories found in the H5 file"
        traj_0 = f['traj_0']
        assert 'obs' in traj_0, "Observations not found in trajectory"
        assert 'actions' in traj_0, "Actions not found in trajectory"
        assert 'rewards' in traj_0, "Rewards not found in trajectory"

def load_from_h5(h5_file: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None, encoding: dict | None = None):
    with h5py.File(h5_file, 'r') as f:
        trajectories = list(f.keys())
        if episodes:
            trajectories = [f[f"traj_{ep}"] for ep in episodes]
        else:
            trajectories = [f[k] for k in filter(lambda k: k.startswith("traj_"), f.keys())]

        ep_dicts = []
        for ep_idx, traj in tqdm(enumerate(trajectories), total=len(trajectories)):
            
            # Load observations, actions, and rewards
            obs = np.concatenate([traj['obs/agent/qpos'], traj['obs/agent/qvel']], axis=1)
            actions = traj['actions'][:]
            rewards = traj['rewards'][:]
            env_state = np.concatenate([
                traj['env_states/actors/cube'],  # pos, quat, vel, ang_vel
                traj['env_states/actors/goal_region'][:, :7],  # pos and quat only
            ], axis=1)
            
            num_frames = len(actions)
            
            ep_dict = {}
            ep_dict["observation.state"] = torch.from_numpy(obs[:-1]).float()
            ep_dict["observation.environment_state"] = torch.from_numpy(env_state[:-1]).float()
            ep_dict["action"] = torch.from_numpy(actions).float()
            ep_dict["next.reward"] = torch.from_numpy(rewards).float()
            ep_dict["next.done"] = torch.zeros(num_frames, dtype=torch.bool)
            ep_dict["next.done"][-1] = True
            ep_dict["next.success"] = torch.zeros(num_frames, dtype=torch.bool)
            ep_dict["next.success"][-1] = True

            # Process images if available
            if 'obs/sensor_data/base_camera/rgb' in traj:
                images = traj['obs/sensor_data/base_camera/rgb'][()]
                img_key = "observation.image"
                if video:
                    video_path = process_images_to_video(images, videos_dir, img_key, ep_idx, fps, encoding)
                    ep_dict["observation.image"] = [{"path": f"videos/{video_path.name}", "timestamp": i / fps} for i in range(num_frames)]
                else:
                    ep_dict["observation.image"] = [PILImage.fromarray(img) for img in images]
            ep_dict["episode_index"] = torch.full((num_frames,), ep_idx, dtype=torch.int64)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

            ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict

def process_images_to_video(images, videos_dir, key, ep_idx, fps, encoding=None):
    # Create a temporary directory to store PNG images
    tmp_imgs_dir = videos_dir / "tmp_images"
    tmp_imgs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save images as PNG files in the temporary directory
        for i, img in enumerate(images):
            img_path = tmp_imgs_dir / f"{i:06d}.png"
            PILImage.fromarray(img).save(img_path)
        
        # Generate the video path
        video_path = videos_dir / f"{key}_episode_{ep_idx:06d}.mp4"
        
        # Use encode_video_frames to generate the video
        encode_video_frames(tmp_imgs_dir, video_path, fps, crf=None, img_format="png", **(encoding or {}))
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(tmp_imgs_dir)
    
    return video_path

def concatenate_episodes(ep_dicts):
    concatenated = {}
    for key in ep_dicts[0].keys():
        if isinstance(ep_dicts[0][key], torch.Tensor):
            concatenated[key] = torch.cat([ep[key] for ep in ep_dicts])
        elif isinstance(ep_dicts[0][key], list):
            concatenated[key] = sum((ep[key] for ep in ep_dicts), [])
    return concatenated

def to_hf_dataset(data_dict, video):
    features = {
        "observation.state": Sequence(feature=Value(dtype="float32", id=None)),
        "observation.environment_state": Sequence(feature=Value(dtype="float32", id=None)),
        "action": Sequence(feature=Value(dtype="float32", id=None)),
        "next.reward": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        "next.success": Value(dtype="bool", id=None),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "index": Value(dtype="int64", id=None),
    }

    if "observation.image" in data_dict:
        if video:
            features["observation.image"] = VideoFrame()
        else:
            features["observation.image"] = Image()

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    h5_file_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    h5_file = h5_file_dir / "trajectory.rgb.pd_joint_pos.cpu.h5"
    check_format(h5_file)

    if fps is None:
        fps = 30  # Default to 30 FPS if not specified

    data_dict = load_from_h5(h5_file, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = encoding or {}

    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    import lerobot
    lerobot_root = Path(lerobot.__path__[0]).parent
    h5_file_dir = lerobot_root / "data/maniskill_raw/PushCube-v1/motionplanning"
    videos_dir = lerobot_root / "data/maniskill_raw/PushCube-v1/motionplanning/videos"
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(h5_file_dir, videos_dir, episodes=[0, 1, 2])
    print(hf_dataset)
    print(episode_data_index)
    print(info)