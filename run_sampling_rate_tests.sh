#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 

for MODEL in "$@"
#for MODEL in lstm \
#             scaledodenetmidpoint \
#             scaledodenetfe \
#             scaledodenetfe \

do
  MODEL_PATH="diode2_clipper/runs/${MODEL}"
  MODEL_CHECKPOINT=`basename $MODEL_PATH`

  for SAMPLING_RATE in 22050 48000 192000
  do
      python diode2_clipper/test.py -cf "${MODEL_PATH}/args.json" --epochs 1 \
      --checkpoint $MODEL_CHECKPOINT --name Test$SAMPLING_RATE \
      --test_sampling_rate $SAMPLING_RATE  --best_validation
  done
done
