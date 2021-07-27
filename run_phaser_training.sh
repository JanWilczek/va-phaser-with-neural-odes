#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 1000 --learn_rate 0.001 --name NewDataset
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.01 --one_cycle_lr 0.02 --teacher_forcing bernoulli --name ResIntNetNewDataset
CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.02 --teacher_forcing always --name ODENet2NewDataset6Layers

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 128 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.01 --teacher_forcing always --weight_decay 0.00001 --dataset_name FameSweetToneOffNoFb --hidden_size 50 --nonlinearity SELU --validate_every 5 --state_size 18
 