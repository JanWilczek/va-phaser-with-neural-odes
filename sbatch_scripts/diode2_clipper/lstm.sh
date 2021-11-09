#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2010M
#SBATCH --output=logs/diode2_clipper/lstm-%j.txt

HIDDEN_SIZE=32
TEST_SAMPLING_RATE=22050

set -x
srun --gres=gpu:1 python diode2_clipper/main.py --method LSTM --epochs 1 --batch_size 64 --learn_rate 0.0001 --init_len 1000 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --hidden_size $HIDDEN_SIZE --dataset_name diode2clip --validate_every 10 --loss_function ESR_DC_prefilter --checkpoint November03_16-24-41_gpu10.int.triton.aalto.fi_November02_13-16-07_gpu9.int.triton.aalto.fi_Hidden32_CONT --name "Hidden${HIDDEN_SIZE}Test${TEST_SAMPLING_RATE}Hz" --test_sampling_rate $TEST_SAMPLING_RATE
