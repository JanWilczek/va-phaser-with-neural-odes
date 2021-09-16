#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method LSTM --batch_size 64 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 1000 --learn_rate 0.001 --dataset_name FameSweetToneOffNoFb --name L1_STFT --loss_function L1_STFT --validate_every 10
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.01 --one_cycle_lr 0.02 --teacher_forcing bernoulli --name ResIntNetNewDataset
CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.02 --teacher_forcing always --name ODENet2NewDataset6Layers

CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --hidden_size 100 --nonlinearity SELU --validate_every 10 --state_size 1 --name L1_STFT --loss_function L1_STFT --checkpoint August09_11-52-35_axel_L1_STFT

CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --hidden_size 30 --nonlinearity SELU --validate_every 10 --state_size 1 --name L1_STFT_DerivativeMLP2 --loss_function L1_STFT

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.0005 --cyclic_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFbAllpassStates --state_size 11 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function ESR_DC_prefilter --weight_decay 0.0000001
 
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 36 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function L2_STFT --weight_decay 0.0000001 --name L2_STFT_DerivativeMLP2

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.0005 --cyclic_lr 0.003 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 36 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance
