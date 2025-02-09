from datasets import load_dataset
from hydra import compose, initialize_config_dir
from pathlib import Path
import importlib.util
from lerobot.common.datasets.factory import make_dataset
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_dataset_card_to_hub, push_videos_to_hub, create_branch, CODEBASE_VERSION
lerobot_spec = importlib.util.find_spec("lerobot")
if lerobot_spec is None:
    raise ImportError("lerobot package not found")

lerobot_root = Path(lerobot_spec.origin).parent
config_path = lerobot_root / "configs"
with initialize_config_dir(config_dir=str(config_path)):
    #local_dataset_root = "/juno/u/bsud2/multi_task_experts/lerobot/"
    local_dataset_root = "/juno/u/bsud2/lerobot_outputs/"
    #local_dataset_repo_id = "d3il_sorting_suboptimal_trajectories_2_successes_dataset/"
    # local_dataset_repo_id = "pusht_suboptimal_trajectories_2_successes_dataset/"
    local_dataset_repo_id = "d3il_sorting_suboptimal_trajectories_2_labeled_dataset/"
    cfg = compose(
        config_name="default.yaml", overrides=["policy=diffusion_d3il_sorting_state",
                                               "env=d3il_sorting_state",
                                               f"dataset_repo_id={local_dataset_repo_id}", 
                                               f"dataset_root={local_dataset_root}"]
    )
    offline_dataset = make_dataset(cfg, root=cfg.dataset_root)
    repo_id = "bhavnasud/d3il_sorting_suboptimal_trajectories_2_labeled"
    offline_dataset.hf_dataset.push_to_hub(repo_id, revision="main")
    metadata_dir = local_dataset_root + local_dataset_repo_id + "meta_data"
    push_meta_data_to_hub(repo_id, metadata_dir, revision="main")
    push_dataset_card_to_hub(repo_id, revision="main")
    # if video:
    #     push_videos_to_hub(repo_id, videos_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

