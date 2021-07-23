import os
import warnings
import torch
import torch.nn as nn



def format_warning(message, category, *args, **kwargs):
    return f'{category.__name__}: {message}' + os.linesep

warnings.filterwarnings("always", category=RuntimeWarning,
                                   module=__name__)
warnings.formatwarning = format_warning

class ExcitationSecondsLinearInterpolation(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = None
        self.excitation_data = None

    def set_excitation_data(self, time, excitation_data):
        """Set the excitation data to interpolate from.

        Parameters
        ----------
        time : torch.Tensor of shape (N,)
            time indices of data point in excitation_data
        excitation_data : torch.Tensor of shape (N, minibatch_size, features_dimensions...)
            excitation data points to interpolate from
        """
        self.time = time
        self.excitation_data = excitation_data
    
    @property
    def dt(self):
        return self.time[1] - self.time[0] # assume a constant time step

    def forward(self, t):
        """Return interpolated excitation data.

        Parameters
        ----------
        t : scalar
            time in seconds at which to interpolate the data

        Returns
        -------
        torch.Tensor of shape (minibatch_size, features_dimensions...)
            interpolated values of the excitation data at t
        """
        last_sample_id = (t // self.dt).type(torch.long)
        next_sample_id = last_sample_id + 1

        if next_sample_id == 0:
            return self.excitation_data[0]
        elif next_sample_id > self.excitation_data.shape[0] - 1:
            warnings.warn(f'Attempting to acces time index {t} beyond available data in time range [{self.time[0]}, {self.time[-1]}].', category=RuntimeWarning)
            return self.excitation_data[-1]

        last_sample_weight = next_sample_id - (t / self.dt)

        return last_sample_weight * self.excitation_data[last_sample_id] + (1 - last_sample_weight) * self.excitation_data[next_sample_id]
