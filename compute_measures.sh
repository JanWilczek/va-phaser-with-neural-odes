#!/bin/bash
set -x

DIODECLIP_44KHZ_TARGET=diode_clipper/data/test/diodeclip-target.wav
DIODECLIP_22KHZ_TARGET=diode_clipper/data/test/diodeclip22050Hz-target.wav
DIODECLIP_48KHZ_TARGET=diode_clipper/data/test/diodeclip48000Hz-target.wav
DIODECLIP_192KHZ_TARGET=diode_clipper/data/test/diodeclip192000Hz-target.wav
: '
# ODENet Hidden 100 FE
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/July13_07-49-07_axel_ODENet2Hidden100/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/July13_07-49-07_axel_ODENet2Hidden100/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/July13_12-01-25_axel_ODENet2Hidden100Test22kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/July13_12-01-25_axel_ODENet2Hidden100Test22kHz/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/July13_10-53-41_axel_ODENet2Hidden100Test48kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/July13_10-53-41_axel_ODENet2Hidden100Test48kHz/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/July13_11-08-11_axel_ODENet2Hidden100Test192kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/July13_11-08-11_axel_ODENet2Hidden100Test192kHz/test_output.wav

# STN
# python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/stn/May20_07-33-56_axel/test_output.wav
# ./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/stn/May20_07-33-56_axel/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/stn/August09_17-36-02_axel_Test22050/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/stn/August09_17-36-02_axel_Test22050/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/stn/August09_17-39-06_axel_Test48000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/stn/August09_17-39-06_axel_Test48000/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/stn/August09_17-45-45_axel_Test192000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/stn/August09_17-45-45_axel_Test192000/test_output.wav

# LSTM
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/lstm/June30_13-57-49_axel/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/lstm/June30_13-57-49_axel/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/lstm/July13_11-57-04_axel_LSTMTest22kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/lstm/July13_11-57-04_axel_LSTMTest22kHz/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/lstm/July13_09-24-08_axel_LSTMTest48kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/lstm/July13_09-24-08_axel_LSTMTest48kHz/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/lstm/July13_11-59-01_axel_LSTMTest192kHz/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/lstm/July13_11-59-01_axel_LSTMTest192kHz/test_output.wav

# ODENet Hidden 9 FE
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/July16_13-14-04_axel_ODENet2Hidden9/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/July16_13-14-04_axel_ODENet2Hidden9/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/August07_09-35-21_axel_Hidden9Test22050/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/August07_09-35-21_axel_Hidden9Test22050/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/August07_09-42-23_axel_Hidden9Test48000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/August07_09-42-23_axel_Hidden9Test48000/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/forward_euler/August07_10-00-01_axel_Hidden9Test192000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/forward_euler/August07_10-00-01_axel_Hidden9Test192000/test_output.wav

# ODENet Hidden 9 implicit Adams
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/odeint_implicit_adams/August04_12-16-11_axel/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/odeint_implicit_adams/August04_12-16-11_axel/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-31-06_axel_Test22050/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-31-06_axel_Test22050/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-31-54_axel_Test48000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-31-54_axel_Test48000/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-32-43_axel_Test192000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/odeint_implicit_adams/August07_09-32-43_axel_Test192000/test_output.wav

# RINN4
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/resintrk4/July06_12-52-13_axelFinalLossFunctionStep1/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/resintrk4/July06_12-52-13_axelFinalLossFunctionStep1/test_output.wav
python compute_measures.py -c $DIODECLIP_22KHZ_TARGET -e diode_clipper/runs/diodeclip/resintrk4/August07_18-46-02_axel_Test22050/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_22KHZ_TARGET -t diode_clipper/runs/diodeclip/resintrk4/August07_18-46-02_axel_Test22050/test_output.wav
python compute_measures.py -c $DIODECLIP_48KHZ_TARGET -e diode_clipper/runs/diodeclip/resintrk4/August07_19-12-33_axel_Test48000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_48KHZ_TARGET -t diode_clipper/runs/diodeclip/resintrk4/August07_19-12-33_axel_Test48000/test_output.wav
python compute_measures.py -c $DIODECLIP_192KHZ_TARGET -e diode_clipper/runs/diodeclip/resintrk4/August07_20-00-25_axel_Test192000/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_192KHZ_TARGET -t diode_clipper/runs/diodeclip/resintrk4/August07_20-00-25_axel_Test192000/test_output.wav
'

# Numerical solvers
# BDF
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/ode_solver/BDF/June16_12-49-56_DESKTOP-O3T26KF/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/ode_solver/BDF/June16_12-49-56_DESKTOP-O3T26KF/test_output.wav
# RK4
python compute_measures.py -c $DIODECLIP_44KHZ_TARGET -e diode_clipper/runs/diodeclip/ode_solver/RK45/June17_08-34-31_DESKTOP-O3T26KF/test_output.wav
./peaqb-fast/src/peaqb -r $DIODECLIP_44KHZ_TARGET -t diode_clipper/runs/diodeclip/ode_solver/RK45/June17_08-34-31_DESKTOP-O3T26KF/test_output.wav

cat analized >> peaq.txt
rm analized
