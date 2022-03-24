#!/bin/bash

set -x

for sampling_rate in 22050 44100 48000 192000
do
    for filename in ../Paper/Audio\ examples/Diode\ clipper/$sampling_rate/*.flac
    do
        # echo $filename
        python scripts/normalize_audio_file.py --target_loudness -23 "$filename"
    done
done
