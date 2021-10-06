from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz


def plot(signal, output_filepath, sampling_rate):
    # setup_pyplot_for_latex()

    plt.figure()
    plt.plot(signal)
    plt.xlabel('Time [s]')
    plt.show()

def main():
    ap = ArgumentParser()
    ap.add_argument('filepath', help='.wav file to compute STFT from.')
    args = ap.parse_args()
    filepath = Path(args.filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    output_filename = filepath.name[:-4] + '_stft'
    output_path = filepath.parent / output_filename
    plot(data, output_path, fs)

if __name__=='__main__':
    main()