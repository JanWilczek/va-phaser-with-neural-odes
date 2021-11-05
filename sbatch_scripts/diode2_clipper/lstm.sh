#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=3000M
#SBATCH --output=logs/diode2_clipper/lstm-%j.txt

set -x
srun python diode2_clipper/main.py --method LSTM --epochs 2000 --batch_size 64 --learn_rate 0.0001 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --hidden_size 32 --dataset_name diode2clip --validate_every 10 --loss_function ESR_DC_prefilter --checkpoint November02_13-16-07_gpu9.int.triton.aalto.fi --name November02_13-16-07_gpu9.int.triton.aalto.fi_CONT
