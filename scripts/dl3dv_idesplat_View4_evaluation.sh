
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
dataset.roots='[path/to/dl3dv]' \
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_model=checkpoints/idesplat-dl3dv-256x448-randview2-6.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=4 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json \
model.encoder.num_scales=1 \
> idesplat-dl3dv-view4_evaluation.log 2>&1 &



