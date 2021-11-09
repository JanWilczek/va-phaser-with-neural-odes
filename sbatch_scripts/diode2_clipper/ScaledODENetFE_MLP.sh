#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80000M
#SBATCH --output=logs/diode2_clipper/scaledodenetfe-%j.txt

HIDDEN_SIZE=30
set -x

for TEST_SAMPLING_RATE in 22050 48000 192000
do
srun --gres=gpu:1 python diode2_clipper/main.py --method ScaledODENetFE --epochs 1 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.1 --exponential_lr 0.001 --teacher_forcing always --dataset_name diode2clip --hidden_size $HIDDEN_SIZE --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Softsign --derivative_network DerivativeMLPFE --name "Hidden${HIDDEN_SIZE}Test${TEST_SAMPLING_RATE}" --checkpoint November01_16-17-57_gpu31.int.triton.aalto.fi_Hidden30_CONT --test_sampling_rate $TEST_SAMPLING_RATE
done
