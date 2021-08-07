#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 

for SAMPLING_RATE in 22050 48000 192000 
do
    python diode_clipper/main.py --method forward_euler --epochs 300 --batch_size 256 --learn_rate 0.001 --one_cycle_lr 0.02 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --teacher_forcing always --hidden_size 9 --dataset_name diodeclip --checkpoint July16_13-14-04_axel_ODENet2Hidden9 --test_sampling_rate $SAMPLING_RATE --name Hidden9Test$SAMPLING_RATE

    python diode_clipper/main.py --method odeint_implicit_adams --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 600 --learn_rate 0.001 --cyclic_lr 0.01 --nonlinearity SELU --dataset_name diodeclip --teacher_forcing always --validate_every 10 --state_size 1 --hidden_size 9 --checkpoint August04_12-16-11_axel --test_sampling_rate $SAMPLING_RATE --name Test$SAMPLING_RATE
done
