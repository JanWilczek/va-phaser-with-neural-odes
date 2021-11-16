from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, decimate
from common import setup_pyplot_for_latex, save_tikz, magnitude_spectrum


def plot_frequency_alignment(target, ours, theirs, frequencies, output_filepath):
    setup_pyplot_for_latex(font_face='Times', font_size=17)

    plt.figure()
    plt.plot(frequencies, target, linewidth=2)
    plt.plot(frequencies, ours, '--', linewidth=2)
    plt.plot(frequencies, theirs, ':', linewidth=2)
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.xlabel('Frequency [kHz]')
    plt.legend(['Target', 'ODENet30(FE)', 'LSTM16'])
    from_frequency = 2000  # in Hz
    to_frequency = 3000  # in Hz
    plt.yticks([-60, -40, -20, 0])
    plt.xticks([50, 100, 250, 500, 1000, 2000, 2500, 3000, 5000, 10000], ['50', '100', '250', '500', '1', '2', '2.5', '3', '5', '10'])
    plt.xlim([from_frequency, to_frequency])
    plt.ylim([-70, -50])
    plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=600, transparent=True)
    # save_tikz(output_filepath)
    # plt.show()

def preprocess_audio_file(filepath, seconds):
    filepath = Path(filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    if seconds:
        data = data[:int(fs * seconds)]
    spectrum = magnitude_spectrum(data, dB=True, normalize_spectrum=True)
    frequencies = np.fft.rfftfreq(data.shape[0], 1 / fs)
    
    # step = 5
    # spectrum = medfilt(spectrum, step)
    # frequencies = medfilt(frequencies, step)
    # q = 3
    # spectrum = decimate(spectrum, q)
    # frequencies = decimate(frequencies, q)

    return spectrum, frequencies

def main():
    ap = ArgumentParser()
    ap.add_argument('--target', help='.wav file with a fragment of the target signal.')
    ap.add_argument('--ours', help='.wav file with a fragment of the estimated signal with our method.')
    ap.add_argument('--theirs', help='.wav file with a fragment of the estimated signal with their method.')
    ap.add_argument('--seconds', default=None, type=float, help='number of seconds into the file to take.')
    args = ap.parse_args()
    
    target_spectrum, frequencies = preprocess_audio_file(args.target, args.seconds)
    ours_spectrum, frequencies_ours = preprocess_audio_file(args.ours, args.seconds)
    theirs_spectrum, frequencies_theirs = preprocess_audio_file(args.theirs, args.seconds)
    
    assert (frequencies == frequencies_ours).all()
    assert (frequencies == frequencies_theirs).all()
    
    output_filename = 'frequency_alignment_comparison'
    output_path = Path(args.target).parent / output_filename
    plot_frequency_alignment(target_spectrum, ours_spectrum, theirs_spectrum, frequencies, output_path)

if __name__=='__main__':
    main()