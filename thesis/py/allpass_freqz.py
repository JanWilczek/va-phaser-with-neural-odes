import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt


def main():
    fc = 1000
    fs = 44100
    wb = 2 * np.pi * fc / fs
    a1 = - (1 - np.tan(wb/2)) / (1 + np.tan(wb/2))

    w, h = freqz([a1, 1], [1, a1])

    phase_mine = np.unwrap(np.arctan((a1**2 - 1) * np.sin(w) / (2*a1 + (1 + a1**2)*np.cos(w))), period=np.pi)

    phase_kiiski = -w + 2 * np.arctan(a1 * np.sin(w) / (1 + a1 * np.cos(w)))

    phase_chirp = -np.arctan(((1 + a1)** 2) * np.sin(w) / (((1 + a1) ** 2) * np.cos(w) + 2 * a1))
    print(phase_chirp)

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital allpass filter frequency response')
    # ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    # ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g--')
    ax2.plot(w, phase_mine, '--')
    ax2.plot(w, phase_kiiski, '.-')
    ax2.plot(w, phase_chirp, '-*')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
    plt.legend(['freqz', 'mine', 'kiiski', 'chirp'])
    plt.show()

if __name__ == '__main__':
    main()
