#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1000M
#SBATCH --output=logs/diode2_clipper/stn-%j.txt

set -x
srun python diode2_clipper/main.py --method STN --layers_description 3x4x4x4x2 --epochs 400 --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --learn_rate 0.005 --exponential_lr 0.0001 --teacher_forcing bernoulli --dataset_name diode2clip --validate_every 10 --loss_function ESR_DC_prefilter --nonlinearity Tanh
