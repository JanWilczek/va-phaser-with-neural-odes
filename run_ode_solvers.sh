#!/bin/bash
INPUT_SCALING_FACTOR=10
set -x  # Display executed commands
# python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --length-seconds $LENGTH --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length "$(($LENGTH*44100))" --normalize
# python diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length 0 --normalize
# python diode_clipper/diode_ode_numerical.py --method-name implicit_adams --upsample-factor 8 --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length 0 --normalize
python diode_clipper/diode_ode_numerical.py --method-name forward_euler --upsample-factor 38 --input-scaling-factor $INPUT_SCALING_FACTOR --frame-length 0 --normalize
