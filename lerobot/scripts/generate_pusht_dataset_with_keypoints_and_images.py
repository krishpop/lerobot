
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
from hydra import compose, initialize_config_dir
import torch
from lerobot.common.datasets.utils import calculate_episode_data_index, reset_episode_index
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data, push_dataset_card_to_hub, push_dataset_to_hub, push_meta_data_to_hub, push_videos_to_hub
from lerobot.common.datasets.utils import create_branch
from pathlib import Path
import importlib.util

lerobot_spec = importlib.util.find_spec("lerobot")
if lerobot_spec is None:
    raise ImportError("lerobot package not found")

lerobot_root = Path(lerobot_spec.origin).parent
config_path = lerobot_root / "configs"

with initialize_config_dir(config_dir=str(config_path)):
    dataset = LeRobotDataset("lerobot/pusht")
    # print(dataset.hf_dataset.with_format(None))
    keypoints_dataset = LeRobotDataset("lerobot/pusht_keypoints")
    # print(keypoints_dataset.hf_dataset.with_format(None))

    
    # Now add the key from dataset_2 to dataset_1
    def add_key_from_another_dataset(example, idx):
        example['observation.environment_state'] = keypoints_dataset.hf_dataset[idx]['observation.environment_state']
        return example
    dataset.hf_dataset = dataset.hf_dataset.map(add_key_from_another_dataset, with_indices=True)

    info = {
        "fps": 10,
        "video": True,
    }
    stats = compute_stats(dataset, 32, 8)
    hf_dataset = dataset.hf_dataset
    train_dir = f"/juno/u/bsud2/multi_task_experts/lerobot/pusht_with_keypoints_and_images/lerobot/pusht/train"
    metadata_dir = f"/juno/u/bsud2/multi_task_experts/lerobot/pusht_with_keypoints_and_images/lerobot/pusht/meta_data"
    videos_dir = f"/juno/u/bsud2/.cache/huggingface/hub/datasets--lerobot--pusht/snapshots/31a41e121d12821207155735793fb9175d29a5dd/videos"
    hf_dataset.with_format(None).save_to_disk(train_dir)
    save_meta_data(info, stats, dataset.episode_data_index, Path(metadata_dir))
    hf_dataset.push_to_hub("bhavnasud/pusht_keypoints_images", revision="main")
    push_meta_data_to_hub("bhavnasud/pusht_keypoints_images", metadata_dir, revision="main")
    push_dataset_card_to_hub("bhavnasud/pusht_keypoints_images", revision="main")
    push_videos_to_hub("bhavnasud/pusht_keypoints_images", videos_dir, revision="main")
    create_branch("bhavnasud/pusht_keypoints_images", repo_type="dataset", branch=CODEBASE_VERSION)

