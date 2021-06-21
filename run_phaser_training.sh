#!/bin/bash
python phaser/main.py LSTM --batch_size 128 --init_len 0 --up_fr 4096 --val_chunk 22050 --test_chunk 22050 --epochs 2 --learn_rate 0.001
