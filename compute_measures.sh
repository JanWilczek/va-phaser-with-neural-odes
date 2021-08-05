#!/bin/bash
set -x
# ODENet
python compute_measures.py -c diode_clipper/data/test/diodeclip-target.wav -e diode_clipper/runs/diodeclip/forward_euler/July13_07-49-07_axel_ODENet2Hidden100/test_output.wav
python compute_measures.py -c diode_clipper/data/test/diodeclip22050Hz-target.wav -e diode_clipper/runs/diodeclip/forward_euler/July13_12-01-25_axel_ODENet2Hidden100Test22kHz/test_output.wav
python compute_measures.py -c diode_clipper/data/test/diodeclip48000Hz-target.wav -e diode_clipper/runs/diodeclip/forward_euler/July13_10-53-41_axel_ODENet2Hidden100Test48kHz/test_output.wav
python compute_measures.py -c diode_clipper/data/test/diodeclip192000Hz-target.wav -e diode_clipper/runs/diodeclip/forward_euler/July13_11-08-11_axel_ODENet2Hidden100Test192kHz/test_output.wav
