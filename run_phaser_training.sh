#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method LSTM --batch_size 64 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 1000 --learn_rate 0.001 --dataset_name FameSweetToneOffNoFb --name L1_STFT --loss_function L1_STFT --validate_every 10
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.01 --one_cycle_lr 0.02 --teacher_forcing bernoulli --name ResIntNetNewDataset
CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.02 --teacher_forcing always --name ODENet2NewDataset6Layers

CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --hidden_size 100 --nonlinearity SELU --validate_every 10 --state_size 1 --name L1_STFT --loss_function L1_STFT --checkpoint August09_11-52-35_axel_L1_STFT

CUDA_VISIBLE_DEVICES=0 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --hidden_size 30 --nonlinearity SELU --validate_every 10 --state_size 1 --name L1_STFT_DerivativeMLP2 --loss_function L1_STFT
#--------------------------------------------------------
# FameSweetDryWet dataset (for dummies)
#--------------------------------------------------------
#--------------- TO RUN -----------------------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --teacher_forcing always --dataset_name FameSweetDryWet --state_size 1 --hidden_size 10 --nonlinearity Identity --validate_every 10 --loss_function ESRLoss --derivative_network DerivativeMLP --name state1
#------------------------ ALREADY RUN -----------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 100 --teacher_forcing always --dataset_name FameSweetDryWet --state_size 1 --hidden_size 10 --nonlinearity Identity --validate_every 10 --loss_function ESRLoss --derivative_network SingleLinearLayer --name SingleLinearLayer

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --teacher_forcing always --dataset_name FameSweetDryWet --state_size 1 --hidden_size 10 --nonlinearity Identity --validate_every 10 --loss_function ESRLoss --derivative_network ScaledSingleLinearLayer --name ScaledSingleLinearLayer

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --dataset_name FameSweetDryWet --hidden_size 16 --validate_every 10 --loss_function ESRLoss

#--------------------------------------------------------
# PinkNoise dataset
#--------------------------------------------------------
#--------------- TO RUN -----------------------------------------
#------------------------ ALREADY RUN -----------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --teacher_forcing always --dataset_name PinkNoise --state_size 1 --hidden_size 15 --nonlinearity Tanh --validate_every 10 --loss_function log_spectral_distance --derivative_network DerivativeMLP --name state1

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --teacher_forcing always --dataset_name PinkNoise --state_size 11 --hidden_size 15 --nonlinearity Tanh --validate_every 10 --loss_function log_spectral_distance --derivative_network DerivativeMLP --name state11

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --dataset_name PinkNoise --hidden_size 16 --validate_every 10 --loss_function log_spectral_distance
#--------------------------------------------------------
# Allpass states
#--------------------------------------------------------
#--------------- TO RUN -----------------------------------------
#--------------------------------------------------------
#--------------- ALREADY RUN -----------------------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --teacher_forcing always --dataset_name FameSweetToneOffNoFbAllpassStates --state_size 11 --hidden_size 9 --nonlinearity Identity --validate_every 10 --loss_function ESRLoss --weight_decay 0.0000001 --derivative_network DerivativeMLP

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.0005 --cyclic_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFbAllpassStates --state_size 11 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function ESRLoss --weight_decay 0.0000001 --derivative_network DerivativeMLP --name DerivativeMLP

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.0005 --cyclic_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFbAllpassStates --state_size 11 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function ESRLoss --weight_decay 0.0000001 --derivative_network DerivativeMLP2 --name DerivativeMLP2

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --dataset_name FameSweetToneOffNoFbAllpassStates --state_size 11 --hidden_size 16 --validate_every 10 --loss_function ESRLoss
#--------------------------------------------------------
#--------------------------------------------------------
# Time-frequency domain loss functions
#--------------------------------------------------------

#--------------- TO RUN -----------------------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 18 --hidden_size 30 --nonlinearity Softsign --validate_every 10 --loss_function DC_log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_DC_DerivativeMLP1_Softsign_CorrectedFE_ExponentialLR --derivative_network DerivativeMLP

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing bernoulli --dataset_name FameSweetToneOffNoFb --state_size 18 --hidden_size 30 --nonlinearity Softsign --validate_every 10 --loss_function DC_log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_DC_DerivativeMLP1_Softsign_CorrectedFE_ExponentialLR --derivative_network DerivativeMLP
#--------------------------------------------------------
#--------------- ALREADY RUN -----------------------------------------
 
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.001 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 36 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function L2_STFT --weight_decay 0.0000001 --name L2_STFT_DerivativeMLP2

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1200 --learn_rate 0.0005 --one_cycle_lr 0.003 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 36 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function DC_log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_DC

#--------------------------------------------------------
 
#--------------------------------------------------------
# Test run for thesis
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1 --learn_rate 0.0005 --one_cycle_lr 0.003 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 36 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance --checkpoint September17_11-38-59_axel_log_spectral_distance --derivative_network DerivativeMLP2
#--------------------------------------------------------

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1000 --learn_rate 0.001 --teacher_forcing never --dataset_name FameSweetToneOffNoFb --hidden_size 16 --validate_every 10 --loss_function L2_STFT --weight_decay 0.0000001 --name L2_STFT

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method LSTM --batch_size 64 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 1000 --learn_rate 0.001 --teacher_forcing never --dataset_name FameSweetToneOffNoFb --hidden_size 16 --validate_every 10 --loss_function DC_log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_DC
#--------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method ResIntRK4 --batch_size 64 --up_fr 2048 --val_chunk 0 --test_chunk 0 --epochs 1000 --learn_rate 0.001 --teacher_forcing bernoulli --dataset_name FameSweetToneOffNoFb --hidden_size 16 --validate_every 10 --loss_function log_spectral_distance --weight_decay 0.0000001

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 201 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 18 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_state18 --checkpoint September18_09-37-43_axel_log_spectral_distance_state18

CUDA_VISIBLE_DEVICES=1 python phaser/main.py --method forward_euler --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 201 --learn_rate 0.001 --one_cycle_lr 0.005 --teacher_forcing always --dataset_name FameSweetToneOffNoFb --state_size 1 --hidden_size 30 --nonlinearity SELU --validate_every 10 --loss_function log_spectral_distance --weight_decay 0.0000001 --name log_spectral_distance_state18 --checkpoint September17_21-59-25_axel_log_spectral_distance_state1
