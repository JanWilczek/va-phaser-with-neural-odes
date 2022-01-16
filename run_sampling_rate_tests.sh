#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 

for SAMPLING_RATE in 22050 48000 192000 
do
    python diode2_clipper/test.py --method STN --epochs 1 --batch_size 25 --learn_rate 0.000000005 --up_fr 100 --val_chunk 22050 --checkpoint November29_09-51-15_gpu22.int.triton.aalto.fi_Hidden30 --name Test$SAMPLING_RATE --test_sampling_rate $SAMPLING_RATE --dataset_name diode2clip --nonlinearity Tanh --validate_every 10 --loss_function ESRLoss --layers_description 3x30x30x2 --teacher_forcing bernoulli --weight_decay 0 --best_validation

    # python diode_clipper/test.py --method LSTM --batch_size 40 --init_len 1000 --epochs 200 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --dataset_name diodeclip --hidden_size 8 --checkpoint June30_13-57-49_axel --test_sampling_rate $SAMPLING_RATE --name Test$SAMPLING_RATE
    
    # python diode_clipper/test.py --method STN --batch_size 256 --epochs 140 --up_fr 22050 --val_chunk 22050 --test_chunk 0 --learn_rate 0.001 --teacher_forcing bernoulli --dataset_name diodeclip --checkpoint May20_07-33-56_axel  --test_sampling_rate $SAMPLING_RATE --name Test$SAMPLING_RATE

    # python diode_clipper/test.py --method forward_euler --epochs 300 --batch_size 256 --learn_rate 0.001 --one_cycle_lr 0.02 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --teacher_forcing always --hidden_size 100 --dataset_name diodeclip --checkpoint July13_07-49-07_axel_ODENet2Hidden100 --test_sampling_rate $SAMPLING_RATE --name Hidden100Test$SAMPLING_RATE

    # python diode_clipper/test.py --method forward_euler --epochs 300 --batch_size 256 --learn_rate 0.001 --one_cycle_lr 0.02 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 0 --teacher_forcing always --hidden_size 9 --dataset_name diodeclip --checkpoint July16_13-14-04_axel_ODENet2Hidden9 --test_sampling_rate $SAMPLING_RATE --name Hidden9Test$SAMPLING_RATE

    # python diode_clipper/test.py --method odeint_implicit_adams --batch_size 256 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 600 --learn_rate 0.001 --cyclic_lr 0.01 --nonlinearity SELU --dataset_name diodeclip --teacher_forcing always --validate_every 10 --state_size 1 --hidden_size 9 --checkpoint August04_12-16-11_axel --test_sampling_rate $SAMPLING_RATE --name Test$SAMPLING_RATE

    # python diode_clipper/test.py --method ResIntRK4 --hidden_size 6 --epochs 600 --batch_size 512 --learn_rate 0.001 --one_cycle_lr 0.02 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --dataset_name diodeclip --checkpoint July06_12-52-13_axelFinalLossFunctionStep1  --test_sampling_rate $SAMPLING_RATE --name Test$SAMPLING_RATE
done
