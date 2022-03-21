#!/bin/bash

set -x

for filename in ../Paper/Audio\ examples/Diode\ clipper\ 2/22050/*.flac
do
#  echo $filename
    python scripts/normalize_audio_file.py --target_loudness -12 "$filename"
done
