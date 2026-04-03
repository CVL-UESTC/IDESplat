
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=acid \
dataset.test_chunk_interval=1 \
dataset.roots='[path/to/acid]' \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_model=checkpoints/idesplat-re10k-256x256-view2.ckpt \
mode=test \
dataset/view_sampler=evaluation \
> idesplat-re10k-for-acid_evaluation.log 2>&1 &



