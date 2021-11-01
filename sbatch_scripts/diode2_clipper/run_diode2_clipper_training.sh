#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1500M
#SBATCH --output=logs/diode2_clipper/%j.txt

# srun python diode2_clipper/main.py --method LSTM --batch_size 64 --init_len 1000 --epochs 1000 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --dataset_name diode2clip --hidden_size 32 --loss_function ESR_DC_prefilter --validate_every 10

# srun python diode2_clipper/main.py --method forward_euler --epochs 1200 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size 30 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP

