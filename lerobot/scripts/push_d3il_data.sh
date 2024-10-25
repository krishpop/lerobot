python lerobot/scripts/push_dataset_to_hub.py --raw-dir /home/ksrini/Temporary_D3IL/environments/dataset/data/stacking/vision_data \
	--raw-format stacking_pkl --repo-id krishpop/d3il_stacking \
        --local-dir data/krishpop/d3il_stacking \
        --batch-size 128 --force-override 1

python lerobot/scripts/push_dataset_to_hub.py --raw-dir /home/ksrini/Temporary_D3IL/environments/dataset/data/sorting/2_boxes \
	--raw-format sorting_pkl --repo-id krishpop/d3il_sorting_2boxes \
        --local-dir data/krishpop/d3il_sorting_2boxes --num-boxes 2 \
        --batch-size 128 --force-override 1

python lerobot/scripts/push_dataset_to_hub.py --raw-dir /home/ksrini/Temporary_D3IL/environments/dataset/data/sorting/4_boxes \
	--raw-format sorting_pkl --repo-id krishpop/d3il_sorting_4boxes \
        --local-dir data/krishpop/d3il_sorting_4boxes --num-boxes 4 \
        --batch-size 128 --force-override 1

python lerobot/scripts/push_dataset_to_hub.py --raw-dir /home/ksrini/Temporary_D3IL/environments/dataset/data/sorting/6_boxes \
	--raw-format sorting_pkl --repo-id krishpop/d3il_sorting_6boxes \
        --local-dir data/krishpop/d3il_sorting_6boxes --num-boxes 6 \
        --batch-size 128 --force-override 1
