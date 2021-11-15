import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import tikzplotlib


def setup_pyplot_for_latex():
    # Use LaTeX font to save the figures in the .png format
    # (they are too big for a tikzfigure)
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams.update({'font.size': 12})

def save_tikz(filename):
    tikz_filename = append_ending(str(filename), '.tex')
    tikzplotlib.save(tikz_filename)

def save_png(filename):
    tikz_filename = append_ending(str(filename), '.png')
    plt.savefig(tikz_filename, bbox_inches='tight', dpi=400)

def append_ending(filename, ending):
    if not filename.endswith(ending):
        filename += ending
    return filename

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
    