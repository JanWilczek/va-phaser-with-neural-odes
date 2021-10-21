from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from scipy.interpolate import interp1d


def main():
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

    output_filename = 'diode2clip-target.wav'
    output_filepath = Path(args.filepath).parent / output_filename

    channels = np.array([left_channel, right_channel]).transpose()
    assert channels.shape[1] == 2

    fs = 44100
    interpolation = interp1d(np.array(time_channel) * fs, channels, axis=0)

    length_seconds = 340
    time_samples = np.arange(0, fs * length_seconds, dtype=np.float32)

    interpolated_channels = interpolation(time_samples)
    assert interpolated_channels.shape[1] == 2

    write(output_filepath, fs, interpolated_channels)


if __name__ == "__main__":
    main()
