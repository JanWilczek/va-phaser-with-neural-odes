from argparse import ArgumentParser
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz, save_png


def amplitude2dB(signal):
    """Convert the linear amplitude to an amplitude in decibels."""
    return 20 * np.log10(np.maximum(signal, 1e-7))


def dB2amplitude(signal):
    """Convert the values in the signal from values in decibels to a linear amplitude."""
    return np.power(10, signal / 20)


def normalized(signal):
    """Return rescaled signal so that it is in the [-1,1] range."""
    return signal / np.amax(np.abs(signal))


def magnitude_spectrum(time_signal, dB=True, normalize_spectrum=False):
    """Return magnitude spectrum of the real-valued time_signal.
    If dB is True, convert to decibel scale.
    Do not return the reflected part of the spectrum."""
    spectrum = np.fft.rfft(time_signal, axis=0)
    magnitude = np.abs(spectrum)
    if normalize_spectrum:
        magnitude = normalized(magnitude)
    if dB:
        return amplitude2dB(magnitude)
    return magnitude

def plot_spectrum(data, axis, sampling_rate):
    spectrum = magnitude_spectrum(data, dB=True, normalize_spectrum=True)
    frequencies = np.fft.rfftfreq(data.shape[0], 1 / sampling_rate)

    # step = 11
    # spectrum = medfilt(spectrum, step)
    # frequencies = medfilt(frequencies, step)
    
    axis.semilogx(frequencies, spectrum, 'C2', linewidth=2)
    axis.set_ylabel('Magnitude [dB]')
    axis.grid()
    axis.set_ylim([-80, 0])
    axis.set_yticks([-60, -40, -20, 0])

def main():
    setup_pyplot_for_latex()
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    for i, filepath in enumerate(['thesis/figures/tikz/diode_clipper_aliasing/target.wav',
                    'thesis/figures/tikz/diode_clipper_aliasing/ODENetFE9192000Hz.wav',
                    'thesis/figures/tikz/diode_clipper_aliasing/ODENetFE922050Hz.wav']):
        filepath = Path(filepath)
        data, fs = sf.read(filepath, always_2d=True)
        data = data[:, 0] # Take just the first channel of audio

        plot_spectrum(data, axs[i], fs)
        if i == 2:
            plt.xlabel('Frequency [Hz]')
    from_frequency = 20  # in Hz
    to_frequency = 22050 // 2  # in Hz
    plt.xlim([from_frequency, to_frequency])
    axs[2].set_yticks([-80, -60, -40, -20, 0])
    plt.xticks([50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000], ['50', '100', '250', '500', '1k', '2k', '3k', '5k', '10k'])

    output_filename = 'diode_clipper_aliasing.tex'
    output_path = filepath.parent / output_filename
    save_png(output_path)
    plt.show()
   

if __name__ == '__main__':
    main()
