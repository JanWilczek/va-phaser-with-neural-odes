#!/bin/bash
LENGTH=0
NORMALIZE=--normalize
INPUT_SCALING_FACTOR=5
FRAME_LENGTH=128
set -x  # Display executed commands
python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length $FRAME_LENGTH $NORMALIZE
python diode_clipper/diode_ode_numerical.py --method-name forward_euler --upsample-factor 38 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length $FRAME_LENGTH $NORMALIZE
