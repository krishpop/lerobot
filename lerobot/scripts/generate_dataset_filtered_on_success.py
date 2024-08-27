
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from hydra import compose, initialize_config_dir
import torch
from lerobot.common.datasets.utils import calculate_episode_data_index, reset_episode_index
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from pathlib import Path
import importlib.util

lerobot_spec = importlib.util.find_spec("lerobot")
if lerobot_spec is None:
    raise ImportError("lerobot package not found")

lerobot_root = Path(lerobot_spec.origin).parent
config_path = lerobot_root / "configs"

for dataset_root in ["pusht_dataset_scale_1", "pusht_dataset_scale_0.5", "pusht_dataset_scale_2"]:
    print(dataset_root)
    with initialize_config_dir(config_dir=str(config_path)):
        cfg = compose(
            config_name="default.yaml", overrides=["policy=vqbet", "env=pusht"]
        )
    offline_dataset = make_dataset(cfg, root="../../" + dataset_root)
    hf_dataset = offline_dataset.hf_dataset
    episode_data_index = offline_dataset.episode_data_index
    episode_indices = torch.stack(offline_dataset.hf_dataset.filter(lambda x: x['next.reward'] > 0.95)['episode_index']).unique()
    filtered_hf_dataset = offline_dataset.hf_dataset.filter(lambda x: x['episode_index'].item() in episode_indices)
    filtered_episode_data_index = calculate_episode_data_index(filtered_hf_dataset)
    filtered_hf_dataset = reset_episode_index(filtered_hf_dataset)

    info = {
        "fps": 4,
        "video": False,
    }

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="lerobot/pusht",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir="mmlfd_videos",
    )
    stats = compute_stats(lerobot_dataset, 32, 8)
    filtered_hf_dataset = filtered_hf_dataset.with_format(None)
    filtered_hf_dataset.save_to_disk(f"/juno/u/bsud2/multi_task_experts/lerobot/{dataset_root}_successes/lerobot/pusht/train")
    save_meta_data(info, stats, filtered_episode_data_index, Path(f"/juno/u/bsud2/multi_task_experts/lerobot/{dataset_root}_successes/lerobot/pusht/meta_data"))