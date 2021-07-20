import warnings
import torch
import torch.nn as nn


class ExcitationSecondsLinearInterpolation(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = None
        self.excitation_data = None

    def set_excitation_data(self, time, excitation_data):
        self.time = time
        self.excitation_data = excitation_data
    
    @property
    def dt(self):
        return self.time[1] - self.time[0] # assume a constant time step

    def forward(self, t):
        last_sample_id = (t // self.dt).type(torch.long)
        next_sample_id = last_sample_id + 1

        if next_sample_id == 0:
            return self.excitation_data[0]
        elif next_sample_id > self.excitation_data.shape[0] - 1:
            warnings.warn("Attempting to acces time index beyond available data.")
            return self.excitation_data[-1]

        last_sample_weight = next_sample_id - (t / self.dt)

        return last_sample_weight * self.excitation_data[last_sample_id] + (1 - last_sample_weight) * self.excitation_data[next_sample_id]
