import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd

class DigitalPhaser:
    def __init__(self, allpass_count, lfo):
        self.allpass_chain = [AllPassOrder1() for i in range(allpass_count)]
        self.lfo = lfo

    def process(self, x):
        pass

class AllPassOrder1:
    def __init__(self):
        self.x_h = 0.0

    def process(self, x, Wc):
        """Apply the first order allpass filter to the signal buffer x.
        Does not allocate new arrays.

        Parameters
        ----------
        x : array-like
            signal buffer to be filtered
        Wc : float
            normalized cutoff frequency in range [0, 1],
            where 0 corresponds to DC and 1 to the Nyquist frequency
        """
        c = (np.tan(np.pi * Wc / 2) - 1) / (np.tan(np.pi * Wc / 2) + 1)
        x = np.asarray(x)
        for n in range(x.shape[0]):
            x_h = x[n] - c * self.x_h
            x[n] = c * x_h + self.x_h
            self.x_h = x_h
        return x


def dB2linear(val):
    return np.power(10, val / 20)

def main():
    f = 220
    fs = 44100
    T = 4
    t = np.arange(0, T, 1/fs)
    sine = np.sin(2 * np.pi * f * t)
    saw = (t % (1/f)) * 2 * f - 1.0
    in_time = 0.01
    envelope = np.ones_like(saw)
    envelope_start = - 0.5 * np.cos(np.pi * t[:int(in_time * fs)] / in_time) + 0.5
    envelope[:int(in_time * fs)] = envelope_start
    envelope[-int(in_time * fs):] = np.flip(envelope_start, axis=0)
    volume_dB = -20
    channel = np.multiply(saw, envelope) * dB2linear(volume_dB)
    output = np.stack((channel, channel)) # TODO

    sd.default.device = 14
    sd.play(output.T, fs)
    sd.wait()

if __name__=='__main__':
    main()

