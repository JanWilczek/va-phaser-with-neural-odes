from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from scipy.interpolate import interp1d


def main():
    """
    This program does the following:

    1. Reads the SPICE simulation data from a .txt file (time, channel 1, and channel 2).
    2. Interpolates the data to be at 44100 sampling rate. 
       It does so for the first length_seconds seconds of the signals.
    3. Saves two-channel data with scipy.io.write with as the 32-bit floating point format.   
    """
    ap = ArgumentParser()
    ap.add_argument('filepath', help='Path to the .txt file with SPICE results.')
    args = ap.parse_args()

    time_channel = []
    left_channel = []
    right_channel = []

    with open(args.filepath, 'r') as f:
        line = f.readline() # skip header
        line = f.readline()

        while line:
            values = line.split()
            
            time = float(values[0])
            c1_voltage = float(values[1])
            c2_voltage = float(values[2])

            time_channel.append(time)
            left_channel.append(c1_voltage)
            right_channel.append(c2_voltage)
            
            line = f.readline()

    output_filename = 'diode2clip-target-unshortened.wav'
    output_filepath = Path(args.filepath).parent / output_filename

    channels = np.array([left_channel, right_channel]).transpose()
    assert channels.shape[1] == 2

    fs = 192000
    interpolation = interp1d(np.array(time_channel) * fs, channels, kind='cubic', axis=0)

    length_seconds = 340
    time_samples = np.arange(0, fs * length_seconds, dtype=np.float32)

    interpolated_channels = interpolation(time_samples)
    assert interpolated_channels.shape[1] == 2

    write(output_filepath, fs, interpolated_channels)


if __name__ == "__main__":
    main()
