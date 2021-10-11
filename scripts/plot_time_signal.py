from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz


def plot(signal, output_filepath, sampling_rate):
    setup_pyplot_for_latex()
    plt.rcParams.update({'font.size': 21})

    time = np.arange(0, signal.shape[0]) / sampling_rate

    plt.figure()
    plt.plot(time, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=200)
    plt.show()

def main():
    ap = ArgumentParser()
    ap.add_argument('filepath', help='.wav file to display the time signal from.')
    args = ap.parse_args()
    filepath = Path(args.filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    output_filename = filepath.name[:-4] + '_time_signal'
    output_path = filepath.parent / output_filename
    plot(data, output_path, fs)

if __name__=='__main__':
    main()