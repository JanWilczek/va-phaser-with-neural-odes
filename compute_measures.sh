#!/bin/bash
set -x

DIODECLIP_TARGETS=(diode_clipper/data/test/diodeclip-target.wav diode_clipper/data/test/diodeclip22050Hz-target.wav diode_clipper/data/test/diodeclip48000Hz-target.wav diode_clipper/data/test/diodeclip192000Hz-target.wav)
FILES_WITH_RESULT_FILEPATHS=(44100.txt 22050.txt 48000.txt 192000.txt)

for i in {0..3}
do
    for file in `cat ${FILES_WITH_RESULT_FILEPATHS[i]}`
    do
        python compute_measures.py -c ${DIODECLIP_TARGETS[i]} -e $file
    done
done

# ODE solvers
python compute_measures.py -c ${DIODECLIP_TARGETS[0]} -e diode_clipper/runs/diodeclip/ode_solver/BDF/June16_12-49-56_DESKTOP-O3T26KF/test_output.wav
python compute_measures.py -c ${DIODECLIP_TARGETS[0]} -e diode_clipper/runs/diodeclip/ode_solver/RK45/June17_08-34-31_DESKTOP-O3T26KF/test_output.wav
python compute_measures.py -c ${DIODECLIP_TARGETS[0]} -e diode_clipper/runs/diodeclip/ode_solver/forward_euler/June15_13-50-48_DESKTOP-O3T26KF/test_output.wav
