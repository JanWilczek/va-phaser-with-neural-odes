from argparse import ArgumentParser
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz, save_png, plot_spectrum




def main():
    setup_pyplot_for_latex(font_face='Times', font_size=15)
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
    axs[1].set_ylabel('Magnitude [dB]')
    from_frequency = 20  # in Hz
    to_frequency = 22050 // 2  # in Hz
    plt.xlim([from_frequency, to_frequency])
    axs[2].set_yticks([-80, -60, -40, -20, 0])
    plt.xticks([50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000], ['50', '100', '250', '500', '1k', '2k', '3k', '5k', '10k'])

    output_filename = 'diode_clipper_aliasing_dafx'
    output_path = filepath.parent / output_filename
    save_png(output_path)
    # plt.show()
   

if __name__ == '__main__':
    main()
