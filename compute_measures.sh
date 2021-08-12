#!/bin/bash
set -x

DIODECLIP_44KHZ_TARGET=diode_clipper/data/test/diodeclip-target.wav
DIODECLIP_22KHZ_TARGET=diode_clipper/data/test/diodeclip22050Hz-target.wav
DIODECLIP_48KHZ_TARGET=diode_clipper/data/test/diodeclip48000Hz-target.wav
DIODECLIP_192KHZ_TARGET=diode_clipper/data/test/diodeclip192000Hz-target.wav

for file in `cat 44100.txt`
do
    python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e $file
done

for file in `cat 22050.txt`
do
    python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e $file
done

for file in `cat 48000.txt`
do
    python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e $file
done

for file in `cat 19200.txt`
do
    python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e $file
done
