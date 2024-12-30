# nuScenes
python evaluate_forgenvideo.py \
    --config configs/evaluation/r3d3/r3d3_evaluation_nuscenes.yaml \
    --r3d3_weights data/models/r3d3/r3d3_finetuned.ckpt \
    --r3d3_image_size 448 768 \
    --r3d3_init_motion_only \
    --r3d3_dt_inter=0 \
    --r3d3_n_edges_max=72 \
    --prediction_data_path data/pred_data_path