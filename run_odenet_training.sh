#!/bin/bash
EPOCHS=1
CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet euler --batch_size 256 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet dopri5 --batch_size 256 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet implicit_adams --batch_size 256 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py odenet midpoint --batch_size 256 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1 python diode_clipper/diode_benchmark.py trapezoid_rule --batch_size 256 --init_len 0 --up_fr 22050 --val_chunk 22050 --test_chunk 22050 --epochs $EPOCHS --learn_rate 0.001
