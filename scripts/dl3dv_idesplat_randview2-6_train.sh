# finetune on dl3dv, random view 2-6
# resume from the previously pretrained model on re10k
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=10 \
checkpointing.resume=true \
dataset.roots='[../dl3dv]' \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
trainer.max_steps=100000 \
model.encoder.num_scales=1 \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vits \
checkpointing.every_n_train_steps=10000 \
checkpointing.pretrained_model=checkpoints/idesplat-re10k-256x256-view2.ckpt \
output_dir=checkpoints/dl3dv-256x448-IDESplat-randview2-6




