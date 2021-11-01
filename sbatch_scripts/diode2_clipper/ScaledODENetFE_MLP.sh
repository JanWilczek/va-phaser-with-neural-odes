#!/bin/bash
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1000M
#SBATCH --output=logs/diode2_clipper/scaledodenetfe-%j.txt

srun python diode2_clipper/main.py --method ScaledODENetFE --epochs 1200 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size 10 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPFE
