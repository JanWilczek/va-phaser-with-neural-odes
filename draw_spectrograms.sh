#!/bin/bash

set -x

for filename in ../Presentations/Final/audio/phaser/cropped/*_normalized.wav
do
    python -m common.plot_stft $filename
done
