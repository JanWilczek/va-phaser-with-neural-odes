#!/bin/bash
EPOCHS=700
BATCH_SIZE=128
set -x
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet euler --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet dopri5 --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet midpoint --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py trapezoid_rule --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=0 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size 64 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs 1000 --learn_rate 0.001 --checkpoint June19_13-12-48_axel_June14_22-23-06_continued --name _June19_13-12-48_test_chunk_22050
CUDA_VISIBLE_DEVICES=0 python diode_clipper/diode_benchmark.py forward_euler --batch_size 128 --init_len 0 --up_fr 4096 --val_chunk 22050 --test_chunk 22050 --epochs 100 --learn_rate 0.001
CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 22050 --epochs 500 --learn_rate 0.0005 --cyclic_lr 0.01 --name ResIntRK4_cycliclr
CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 22050 --epochs 100 --learn_rate 0.001 --name ResIntRK4
CUDA_VISIBLE_DEVICES=0 python diode_clipper/main.py ResIntRK4 --batch_size 512 --init_len 0 --up_fr 1024 --val_chunk 22050 --test_chunk 0 --epochs 500 --learn_rate 0.0005 --cyclic_lr 0.01 --name ResIntRK4_cycliclr --checkpoint June23_10-20-36_axelResIntRK4_cycliclr
CUDA_VISIBLE_DEVICES=1 python diode_clipper/main.py LSTM --batch_size 40 --init_len 1000 --epochs 200 --learn_rate 0.001 --up_fr 2048 --val_chunk 22050 --test_chunk 0
