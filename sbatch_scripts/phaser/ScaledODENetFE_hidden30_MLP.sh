#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1700M
#SBATCH --output=logs/phaser/scaledodenetfe-%j.txt

set -x
srun python phaser/main.py --method ScaledODENetFE --epochs 1200 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name FameSweetToneOffNoFbAllpassStates --hidden_size 30 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPFE --checkpoint November02_18-06-17_gpu25.int.triton.aalto.fi
