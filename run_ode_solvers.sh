#!/bin/bash
LENGTH=1
NORMALIZE=--normalize
INPUT_SCALING_FACTOR=10
FRAME_LENGTH=128
set -x  # Display executed commands
# python diode_clipper/diode_ode_numerical.py --method-name implicit_adams --upsample-factor 8 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length $FRAME_LENGTH $NORMALIZE
# echo "1 long block of samples"
python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length "$(($LENGTH*44100))" $NORMALIZE
python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length 0 $NORMALIZE
# echo "Many short blocks of samples"
# python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length $FRAME_LENGTH $NORMALIZE
# python diode_clipper/diode_ode_numerical.py --method-name forward_euler --upsample-factor 38 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length $FRAME_LENGTH $NORMALIZE
