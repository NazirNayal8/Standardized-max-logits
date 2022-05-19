#!/usr/bin/env bash
# Example on Cityscapes
python train.py \
   --dataset cityscapes \
   --val_dataset cityscapes \
   --arch network.deepv3.DeepR101V3PlusD_OS8 \
   --city_mode 'train' \
   --lr_schedule poly \
   --lr 0.01 \
   --poly_exp 0.9 \
   --max_cu_epoch 10000 \
   --class_uniform_pct 0.5 \
   --class_uniform_tile 1024 \
   --crop_size 768 \
   --scale_min 0.5 \
   --scale_max 2.0 \
   --rrotate 0 \
   --max_iter 90000 \
   --bs_mult 2 \
   --gblur \
   --color_aug 0.5 \
   --date 0000 \
   --exp r101_os8_base_60K \
   --ckpt ./logs/ \
   --image_mode RGB \
   --wandb_project sml \
   --freeze_scheduler True \
   --wandb_run_name sml_RGB_4b_continue_in_2b \
   --snapshot ./logs/0000/r101_os8_base_60K/05_17_11/best_cityscapes_epoch_79_mean-iu_0.79702.pth \
   --tb_path ./logs/
