from math import pi
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from excitation import ExcitationSeconds, ExcitationSamples, ExcitationSecondsInterpolation0, ExcitationSecondsInterpolation1


def main():
    T = 4 * 2 * pi
    nsteps = 500
    dt = T / nsteps
    frequency = 0.8
    amplitude = 1.0

    t_samples = torch.arange(0, nsteps)
    t_seconds = torch.arange(0, T, dt)
    assert t_samples.shape[0] == t_seconds.shape[0]

    excitation_seconds = ExcitationSeconds(amplitude, frequency)
    excitation_samples = ExcitationSamples(amplitude, frequency, dt)
    excitation_seconds_interpolation_0 = ExcitationSecondsInterpolation0(amplitude, frequency, dt)
    excitation_seconds_interpolation_1 = ExcitationSecondsInterpolation1(amplitude, frequency, dt)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_seconds, excitation_seconds_interpolation_0(t_seconds), '.')
    plt.plot(t_seconds, excitation_seconds(t_seconds), '--')
    plt.plot(t_seconds, excitation_seconds_interpolation_1(t_seconds), '-.')
    plt.legend(['seconds', 'seconds_interpolation_0', 'seconds_interpolation_1'])
    plt.subplot(2, 1, 2)
    plt.plot(t_samples, excitation_samples(t_samples))
    plt.legend(['samples'])
    plt.savefig(Path('harmonic_oscillator', 'excitation_comparison.png'), bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
