from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz, magnitude_spectrum


def plot_frequency_alignment(target, estimated, frequencies, output_filepath):
    # setup_pyplot_for_latex()

    plt.figure()
    plt.semilogx(frequencies, target, linewidth=2)
    plt.semilogx(frequencies, estimated, '--', linewidth=2)
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.ylim([-60, 0])
    plt.xlabel('Frequency [Hz]')
    plt.legend(['Target', 'ODENet30(FE)'])
    from_frequency = 100  # in Hz
    to_frequency = 22050 // 2  # in Hz
    plt.xlim([from_frequency, to_frequency])
    plt.yticks([-60, -40, -20, 0])
    plt.xticks([50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000], ['50', '100', '250', '500', '1k', '2k', '3k', '5k', '10k'])
    # plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=600, transparent=True)
    # save_tikz(output_filepath)
    plt.show()

def preprocess_audio_file(filepath, seconds):
    filepath = Path(filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    if seconds:
        data = data[:int(fs * seconds)]
    spectrum = magnitude_spectrum(data, dB=True, normalize_spectrum=True)
    frequencies = np.fft.rfftfreq(data.shape[0], 1 / fs)

    return spectrum, frequencies

def main():
    ap = ArgumentParser()
    # TODO: Add the possibility of processing arbitrarily many files
    ap.add_argument('--target', '-t', help='.wav file with a fragment of the target signal.')
    ap.add_argument('--estimated', '-e', help='.wav file with a fragment of the estimated signal.')
    ap.add_argument('--seconds', default=None, type=float, help='number of seconds into the file to take.')
    args = ap.parse_args()
    
    target_spectrum, frequencies = preprocess_audio_file(args.target, args.seconds)
    estimated_spectrum, frequencies_estimated = preprocess_audio_file(args.estimated, args.seconds)
    
    assert (frequencies == frequencies_estimated).all()
    
    output_filename = 'frequency_alignment_comparison'
    output_path = Path(args.target).parent / output_filename
    plot_frequency_alignment(target_spectrum, estimated_spectrum, frequencies, output_path)

if __name__=='__main__':
    main()