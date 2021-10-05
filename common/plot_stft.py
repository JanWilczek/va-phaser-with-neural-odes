from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import setup_pyplot_for_latex, save_tikz


def T_coef(time_index, hop_size, sampling_rate):
    return time_index * hop_size / sampling_rate

def F_coef(frequency_index, window_size, sampling_rate):
    return frequency_index * sampling_rate / window_size

def fft_time(nb_frames, hop_length, sample_rate):
    indices = np.arange(0, nb_frames, 1)
    return T_coef(indices, hop_length, sample_rate)

def plot_stft(signal, output_filepath, sampling_rate):
    setup_pyplot_for_latex()
    n_fft = 4096
    hop_length = n_fft // 2
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    log_magnitude_stft = 10 * np.log10(np.maximum(np.abs(stft), 1e-6))
    K, M = log_magnitude_stft.shape
    t = fft_time(M, hop_length, sampling_rate)
    f_max = F_coef(frequency_index=n_fft//2, window_size=n_fft, sampling_rate=sampling_rate)
    bounding_box = (t[0], t[-1], 0, f_max // 1000)

    plt.figure()
    plt.imshow(log_magnitude_stft, origin='lower', aspect='auto', extent=bounding_box)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.colorbar()
    plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=300, transparent=True)
    save_tikz(output_filepath)

def main():
    ap = ArgumentParser()
    ap.add_argument('filepath', help='.wav file to compute STFT from.')
    args = ap.parse_args()
    filepath = Path(args.filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    output_filename = filepath.name[:-4] + '_stft'
    output_path = filepath.parent / output_filename
    plot_stft(data, output_path, fs)

if __name__=='__main__':
    main()