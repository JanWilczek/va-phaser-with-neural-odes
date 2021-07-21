#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchaudio
from CoreAudioML.training import PreEmph


def magnitude_spectrum(signal_tensor, dt=1.0):
    ndarray = signal_tensor.detach().cpu().numpy()
    spectrum = 20 * np.log10(np.abs(np.fft.rfft(ndarray)))
    frequencies = np.fft.rfftfreq(ndarray.shape[0], dt)
    return frequencies, spectrum

def main():
    pink_noise_filepath = Path('test_signals', 'pink_noise_30s.wav')
    data, fs = torchaudio.load(pink_noise_filepath)
    prefilter = PreEmph([1, -0.85])
    
    with torch.no_grad():
        prefilter_input = data.transpose(0, 1)[:, None]
        _, prefiltered_data = prefilter(prefilter_input, prefilter_input)
        prefiltered_data.squeeze_()
        prefiltered_filepath = pink_noise_filepath.parent / f'prefiltered_{pink_noise_filepath.name}'
        torchaudio.save(prefiltered_filepath, prefiltered_data[None, :], fs)
        
        f_input, prefilter_input_spectrum = magnitude_spectrum(prefilter_input.squeeze(), 1/fs)
        f_prefiltered, prefiltered_magnitude_spectrum = magnitude_spectrum(prefiltered_data, 1/fs)

        plt.figure()
        plt.semilogx(f_input, prefilter_input_spectrum, '.')
        plt.semilogx(f_prefiltered, prefiltered_magnitude_spectrum, '--')
        plt.legend(['Pink noise magnitude spectrum', 'Prefiltered pink noise magnitude spectrum'])
        plt.xlim([20, 20000])
        plt.grid()
        prefiltered_magnitude_spectrum_filepath = prefiltered_filepath.parent / prefiltered_filepath.name.replace('wav', 'png')
        plt.savefig(prefiltered_magnitude_spectrum_filepath)

if __name__ == '__main__':
    main()
