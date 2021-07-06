#!/bin/bash
python phaser/main.py LSTM --batch_size 128 --init_len 0 --up_fr 4096 --val_chunk 22050 --test_chunk 22050 --epochs 2 --learn_rate 0.001
CUDA_VISIBLE_DEVICES=1 python phaser/main.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 22050 --epochs 1000 --learn_rate 0.01 --one_cycle_lr 0.1 --checkpoint June24_16-31-32_axel_latent2x12 --name _latent2x12
CUDA_VISIBLE_DEVICES=0 python phaser/main.py forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.02 --name ODENet2
