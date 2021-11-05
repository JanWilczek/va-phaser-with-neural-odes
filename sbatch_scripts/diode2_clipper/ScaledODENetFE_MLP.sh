#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2000M
#SBATCH --output=logs/diode2_clipper/scaledodenetfe-%j.txt

HIDDEN_SIZE=10

set -x
srun python diode2_clipper/main.py --method ScaledODENetFE --epochs 1 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size $HIDDEN_SIZE --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPFE --name Hidden$HIDDEN_SIZE --checkpoint November01_17-47-15_gpu1.int.triton.aalto.fi_Hidden10
