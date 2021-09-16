import torch
import torch.nn as nn
from CoreAudioML import training


def l1_stft(output, target):
    """L1 distance between the complex STFT of output and target."""
    n_fft = min(1024, output.shape[0])
    stft_kwargs = {'return_complex': True, 'n_fft': n_fft, 'window': torch.hann_window(n_fft).to(output.device), 'center': False}
    stft_output = torch.stft(output.squeeze(2).transpose(0, 1), **stft_kwargs)
    stft_target = torch.stft(target.squeeze(2).transpose(0, 1), **stft_kwargs)
    return nn.L1Loss(reduction='mean')(stft_output, stft_target)

def l2_stft(output, target):
    """L2 distance between the complex STFT of output and target."""
    n_fft = min(1024, output.shape[0])
    stft_kwargs = {'return_complex': True, 'n_fft': n_fft, 'window': torch.hann_window(n_fft).to(output.device), 'center': False}
    stft_output = torch.stft(output.squeeze(2).transpose(0, 1), **stft_kwargs)
    stft_target = torch.stft(target.squeeze(2).transpose(0, 1), **stft_kwargs)
    return nn.MSELoss(reduction='mean')(torch.view_as_real(stft_output), torch.view_as_real(stft_target))

def log_spectral_distance(output, target):
    n_fft = min(1024, output.shape[0])
    stft_kwargs = {'return_complex': True, 'n_fft': n_fft, 'window': torch.hann_window(n_fft).to(output.device), 'center': False}
    stft_output = torch.stft(output.squeeze(2).transpose(0, 1), **stft_kwargs)
    stft_target = torch.stft(target.squeeze(2).transpose(0, 1), **stft_kwargs)
    power_output = stft_output.abs() ** 2
    power_target = stft_target.abs() ** 2
    log_spectral_distance_frames = torch.sqrt(torch.mean(torch.square(torch.log(power_target) - torch.log(power_output)), -2))
    return torch.mean(log_spectral_distance_frames) # Average across time and batch dimensions

def get_loss_function(loss_function_name):
    if loss_function_name == 'ESR_DC_prefilter':
        return training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])
    elif loss_function_name.lower() in globals():
        return globals()[loss_function_name.lower()]
    else:
        raise RuntimeError('Invalid loss function name.')
