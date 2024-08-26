
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from hydra import compose, initialize_config_dir
import torch
from datasets import Image
from lerobot.common.datasets.utils import calculate_episode_data_index, reset_episode_index, concatenate_multi_lerobot_dataset
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from pathlib import Path
import importlib.util

lerobot_spec = importlib.util.find_spec("lerobot")
if lerobot_spec is None:
    raise ImportError("lerobot package not found")

lerobot_root = Path(lerobot_spec.origin).parent
config_path = lerobot_root / "configs"

dataset_roots = ["../datasets/pusht/diffusion/step_1225000/2024-08-16_18-22-03", "../datasets/pusht/diffusion/step_1225000/2024-08-16_18-27-05", "../datasets/pusht/diffusion/step_1225000/2024-08-17_09-22-34"]
dataset_root = "../datasets/pusht/diffusion/step_1225000"

with initialize_config_dir(config_dir=str(config_path)):
    cfg = compose(
        config_name="default.yaml", overrides=["policy=tdmpc_pusht", "env=pusht"]
    )

cfg.dataset_root = dataset_roots
offline_dataset = make_dataset(cfg, root=cfg.dataset_root)
offline_dataset = concatenate_multi_lerobot_dataset(offline_dataset)
hf_dataset = offline_dataset.hf_dataset
episode_data_index = offline_dataset.episode_data_index
def filter_successful_episodes(batch):
    return [i for i, rwd in enumerate(batch) if rwd > 0.95]

filtered_dataset = offline_dataset.hf_dataset.filter(
    filter_successful_episodes,
    batched=True,
    batch_size=1000,
    input_columns=['next.reward']
)

filtered_dataset = reset_episode_index(filtered_dataset)
filtered_episode_data_index = calculate_episode_data_index(filtered_dataset)

offline_dataset = LeRobotDataset.from_preloaded(
    repo_id="lerobot/pusht",
    hf_dataset=filtered_dataset,
    episode_data_index=filtered_episode_data_index,
    info=offline_dataset.info,
)
stats = compute_stats(offline_dataset, 32, 8)
hf_dataset = hf_dataset.with_format(None)
hf_dataset.save_to_disk(f"{dataset_root}_successes/lerobot/pusht/train")
save_meta_data(offline_dataset.info, stats, episode_data_index, Path(f"{dataset_root}_successes/lerobot/pusht/meta_data"))