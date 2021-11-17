#!/bin/bash
#SBATCH --time=42:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --constraint='volta'
#SBATCH --output=logs/diode2_clipper/scaledodenetfe-%j.txt

HIDDEN_SIZE=30
METHOD=rk4
set -x

module load vavnode
conda activate vavnode

srun python diode2_clipper/main.py --method ScaledODENet_${METHOD} --epochs 1200 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size $HIDDEN_SIZE --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP --name "Hidden${HIDDEN_SIZE}_${METHOD}"
# srun --gres=gpu:1 --constraint=volta --time=01:00:00 --mem=3000M python diode2_clipper/main.py --method ScaledODENet_rk4 --epochs 400 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size 20 --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLP
