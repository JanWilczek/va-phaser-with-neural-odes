#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2010M
#SBATCH --constraint='volta'
#SBATCH --output=logs/diode2_clipper/lstm-%j.txt

HIDDEN_SIZE=16
TEST_SAMPLING_RATE=44100

set -x
srun python diode2_clipper/main.py --method LSTM --epochs 5 --batch_size 64 --learn_rate 0.0001 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --hidden_size $HIDDEN_SIZE --dataset_name diode2clip --validate_every 1 --loss_function ESR_DC_prefilter

