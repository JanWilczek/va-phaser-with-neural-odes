#!/bin/bash

set -x

for filename in ../Presentations/Final/audio/phaser/cropped/*.wav
do
    python -m scripts.normalize_audio_file $filename --target_loudness -23
done
