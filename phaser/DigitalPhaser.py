import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf


class DigitalPhaser:
    def __init__(self, lfo, allpass_cutoff_frequencies=10*[(250/44100)], allpass_modulation_index=0.2):
        self.allpass_chain = [AllPassOrder1() for i in range(len(allpass_cutoff_frequencies))]
        self.lfo = lfo
        self.allpass_cutoff_frequencies = allpass_cutoff_frequencies
        self.allpass_modulation_index = allpass_modulation_index
        self.wetness = 0.5
        self.x_fb = 0.0
        self.fb = 0.0

    @property
    def allpass_cutoff_frequencies(self):
        return self.__Wc

    @allpass_cutoff_frequencies.setter
    def allpass_cutoff_frequencies(self, value):
        self.__Wc = value

    @property
    def allpass_modulation_index(self):
        return self.__m

    @allpass_modulation_index.setter
    def allpass_modulation_index(self, value):
        self.__m = value

    @property
    def wetness(self):
        return self.__wetness
    
    @wetness.setter
    def wetness(self, value):
        self.dryness = 1.0 - value
        self.__wetness = value

    def process(self, x):
        y = np.zeros_like(x)
        for n in range(x.shape[0]):
            allpass_input = x[n] + self.x_fb * self.fb
            for i, allpass in enumerate(self.allpass_chain):
                Wc_in = self.allpass_cutoff_frequencies[i] * (1 + self.allpass_modulation_index * self.lfo())
                allpass_input = allpass.process(allpass_input, Wc_in)
            self.lfo.increment_phase()
            self.x_fb = allpass_input * self.wetness
            y[n] = self.dryness * x[n] + self.x_fb
        return y

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
        x = np.atleast_1d(x)
        for n in range(x.shape[0]):
            x_h = x[n] - c * self.x_h
            x[n] = c * x_h + self.x_h
            self.x_h = x_h
        return x

class Oscillator:
    def __init__(self, frequency, sample_rate, waveform):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.waveform = waveform
        self.frequency = frequency

    def __call__(self):
        return self.waveform(self.phase)

    @property
    def frequency(self):
        return self.__frequency
    
    @frequency.setter
    def frequency(self, value):
        self.__frequency = value
        self.phase_increment = 2 * np.pi * self.frequency / self.sample_rate

    def increment_phase(self):
        self.phase += self.phase_increment

def dB2linear(val):
    return np.power(10, val / 20)

def phaser_test():
    f = 220
    fs = 44100
    T = 10
    t = np.arange(0, T, 1/fs)
    sine = np.sin(2 * np.pi * f * t)
    saw = (t % (1/f)) * 2 * f - 1.0
    in_time = 0.01
    envelope = np.ones_like(saw)
    envelope_start = - 0.5 * np.cos(np.pi * t[:int(in_time * fs)] / in_time) + 0.5
    envelope[:int(in_time * fs)] = envelope_start
    envelope[-int(in_time * fs):] = np.flip(envelope_start, axis=0)
    volume_dB = -20
    phaser_input = np.multiply(saw, envelope) * dB2linear(volume_dB)
    # channel = phaser_input

    # phaser = DigitalPhaser(Oscillator(0.3, fs, np.sin), allpass_modulation_index=0.9)
    base_frequencies = [85, 250, 600, 900, 2000]
    reversed_frequencies = base_frequencies.copy()
    reversed_frequencies.reverse()
    allpass_cutoff_frequencies = [f/fs for f in base_frequencies + reversed_frequencies]
    phaser = DigitalPhaser(Oscillator(3, fs, np.sin), allpass_cutoff_frequencies, allpass_modulation_index=0.2)
    channel = phaser.process(phaser_input)

    output = np.stack((channel, channel))

    sf.write('sawtooth_phasered.wav', output.T, fs)

    sd.default.device = 14
    sd.play(output.T, fs)
    sd.wait()

def oscillator_test():
    osc = Oscillator(1, 44100, np.sin)
    osc_output = np.zeros((44100,))
    for n in range(osc_output.shape[0]):
        osc_output[n] = osc()
        osc.increment_phase()
    plt.figure()
    plt.plot(osc_output)
    plt.show()

def main():
    # oscillator_test()
    phaser_test()

if __name__=='__main__':
    main()

