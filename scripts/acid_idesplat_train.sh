#!/usr/bin/env bash

# Trained for 300k steps with a batch size of 2 per GPU on 8 GPUs (≥24GB VRAM, e.g., RTX 4090).
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main +experiment=acid \
trainer.max_steps=300000 \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=10 \
dataset.roots='[path/to/acid]' \
checkpointing.resume=true \
model.encoder.lowest_feature_resolution=4 \
checkpointing.every_n_train_steps=10000 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/acid-256x256-IDESplat






