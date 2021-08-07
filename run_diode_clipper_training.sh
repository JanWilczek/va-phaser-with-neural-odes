#!/bin/bash
EPOCHS=700
BATCH_SIZE=128
set -x
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet euler --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet dopri5 --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet midpoint --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py trapezoid_rule --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --one_cycle_lr 0.02 --teacher_forcing always --name ODENet2TFInitialization
CUDA_VISIBLE_DEVICES=0 python diode_clipper/diode_benchmark.py forward_euler --batch_size 128 --init_len 0 --up_fr 4096 --val_chunk 22050 --test_chunk 22050 --epochs 100 --learn_rate 0.001
CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 22050 --epochs 500 --learn_rate 0.0005 --cyclic_lr 0.01 --name ResIntRK4_cycliclr
CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 22050 --epochs 100 --learn_rate 0.001 --name ResIntRK4
CUDA_VISIBLE_DEVICES=0 python diode_clipper/main.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --epochs 500 --learn_rate 0.0005 --cyclic_lr 0.01 --name ResIntRK4_cycliclr --checkpoint June23_10-20-36_axelResIntRK4_cycliclr

CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method LSTM --batch_size 64 --init_len 1000 --epochs 1000 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --dataset_name ht1 --hidden_size 32

CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method STN --batch_size 256 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 300 --learn_rate 0.001 --teacher_forcing bernoulli --dataset_name muff --one_cycle_lr 0.02


CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method odeint_implicit_adams --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 600 --learn_rate 0.001 --cyclic_lr 0.01 --nonlinearity SELU --dataset_name diodeclip --teacher_forcing always --validate_every 10 --state_size 1 --hidden_size 9 --checkpoint July28_14-26-52_axel

python diode_clipper/main.py --method forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 0 --learn_rate 0.001 --dataset_name diodeclip --test_sampling_rate 192000

# reverb-diodeclip
CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method forward_euler --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --epochs 600 --learn_rate 0.001 --cyclic_lr 0.01 --nonlinearity SELU --dataset_name reverb-diodeclip --teacher_forcing always --validate_every 10 --state_size 1 --hidden_size 100

CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py --method LSTM --batch_size 64 --init_len 1000 --epochs 600 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --dataset_name reverb-diodeclip --hidden_size 8 --validate_every 10 
