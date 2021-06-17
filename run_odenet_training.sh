#!/bin/bash
EPOCHS=700
BATCH_SIZE=128
set -x
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet euler --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet dopri5 --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet midpoint --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py trapezoid_rule --batch_size $BATCH_SIZE --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=1 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size 128 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs 200 --learn_rate 0.001 --cyclic_lr
CUDA_VISIBLE_DEVICES=0 python diode_clipper/diode_benchmark.py forward_euler --batch_size 128 --init_len 0 --up_fr 2048 --val_chunk 22050 --test_chunk 22050 --epochs 200 --learn_rate 0.0005 --cyclic_lr 0.01
