# nuScenes
python evaluate.py \
    --config configs/evaluation/dataset_generation/dataset_generation_nuscenes.yaml \
    --r3d3_weights=data/models/r3d3/r3d3_finetuned.ckpt \
    --r3d3_image_size 448 768 \
    --r3d3_n_warmup=5 \
    --r3d3_optm_window=5 \
    --r3d3_corr_impl=lowmem \
    --r3d3_graph_type=droid_slam \
    --training_data_path=/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/MagicDriveDiT-main/data/nuscenes