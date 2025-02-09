
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from hydra import compose, initialize_config_dir
import torch
from lerobot.common.datasets.utils import calculate_episode_data_index, reset_episode_index, load_stats
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from pathlib import Path
import importlib.util

lerobot_spec = importlib.util.find_spec("lerobot")
if lerobot_spec is None:
    raise ImportError("lerobot package not found")

lerobot_root = Path(lerobot_spec.origin).parent
config_path = lerobot_root / "configs"

# for dataset_root in ["pusht_dataset_scale_1", "pusht_dataset_scale_0.5", "pusht_dataset_scale_2"]:
# print(dataset_root)
with initialize_config_dir(config_dir=str(config_path)):
    cfg = compose(
        config_name="default.yaml", overrides=["policy=diffusion_d3il_sorting_state", "env=d3il_sorting_state", "dataset_repo_id=bhavnasud/d3il_sorting_suboptimal_trajectories_2"]
    )
offline_dataset = make_dataset(cfg)
hf_dataset = offline_dataset.hf_dataset
episode_indices = torch.stack(hf_dataset.filter(lambda x: x['next.reward'] > 0.0)['episode_index']).unique()
successful_episode_set = set(episode_indices.tolist())

print("number of successful episodes ", len(successful_episode_set))

# Add "successful_trajectory" field
def add_success_field(entry):
    entry["successful_trajectory"] = entry["episode_index"].item() in successful_episode_set
    return entry

hf_dataset = hf_dataset.map(add_success_field)

info = {
    "fps": 4,
    "video": False,
}

lerobot_dataset = LeRobotDataset.from_preloaded(
    repo_id="bhavnasud/d3il_sorting_suboptimal_trajectories_2",
    hf_dataset=hf_dataset,
    episode_data_index=offline_dataset.episode_data_index,
    info=info,
    # videos_dir="mmlfd_videos",
)
stats = load_stats("bhavnasud/d3il_sorting_suboptimal_trajectories_2", "v1.6", None)
#stats = compute_stats(lerobot_dataset, 32, 8)
hf_dataset = hf_dataset.with_format(None)
hf_dataset.save_to_disk(f"/juno/u/bsud2/lerobot_outputs/d3il_sorting_suboptimal_trajectories_2_labeled_dataset/train")
save_meta_data(info, stats, offline_dataset.episode_data_index, Path(f"/juno/u/bsud2/lerobot_outputs/d3il_sorting_suboptimal_trajectories_2_labeled_dataset/meta_data"))
