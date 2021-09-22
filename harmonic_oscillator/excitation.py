from math import pi
import torch


class ExcitationSeconds:
    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, t):
        return self.amplitude * torch.sin(2 * pi * self.frequency * t)

class ExcitationSamples:
    def __init__(self, amplitude, frequency, dt):
        self.dt = dt
        self.excitation_seconds = ExcitationSeconds(amplitude, frequency)

    def __call__(self, n):
        return self.excitation_seconds(n * self.dt)

class ExcitationSecondsInterpolation0:
    def __init__(self, amplitude, frequency, dt):
        self.excitation_samples = ExcitationSamples(amplitude, frequency, dt)

    def __call__(self, t):
        dt = self.excitation_samples.dt
        sample_id = t // dt
        assert sample_id.dtype == int
        return self.excitation_samples(sample_id)

class ExcitationSecondsInterpolation1:
    def __init__(self, amplitude, frequency, dt):
        self.excitation_samples = ExcitationSamples(amplitude, frequency, dt)

    def __call__(self, t):
        dt = self.excitation_samples.dt
        last_sample_id = t // dt
        next_sample_id = last_sample_id + 1
        last_sample_weight = (next_sample_id - (t / dt))
        return last_sample_weight * self.excitation_samples(last_sample_id) + (1 - last_sample_weight) * self.excitation_samples(next_sample_id)

