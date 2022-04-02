import torch
import torch.nn as nn
from CoreAudioML.training import LossWrapper, ESRLoss, DCLoss


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
    power_output = torch.clamp_min(stft_output.abs() ** 2, 1e-7)
    power_target = torch.clamp_min(stft_target.abs() ** 2, 1e-7)
    log_spectral_distance_frames = torch.sqrt(torch.mean(torch.square(torch.log(power_target) - torch.log(power_output)), -2))
    average_log_spectral_distance = torch.mean(torch.mean(log_spectral_distance_frames, -1)) # Average across time and batch dimensions
    NORMALIZATION_CONSTANT = 3  # Peak average log spectral distance for LSTM at training was 2.85. Thus, 3 seems like a natural normalization constant.
    return average_log_spectral_distance / NORMALIZATION_CONSTANT


def normalized_mean_squared_error(output, target):
    return torch.mean(torch.div(torch.square(output - target), stabilize(torch.square(target))))


def stabilize(tensor):
    return torch.maximum(tensor, torch.ones_like(tensor) * 1e-5)


def signal_to_distortion_ratio(output, target):
    return 10 * torch.log10(torch.div(torch.square(torch.linalg.norm(target, ord=2)),
                                      stabilize(torch.square(torch.linalg.norm(output - target, ord=2)))))


def to_db(signal):
    return 10 * torch.log10(stabilize(signal))


def get_loss_function(loss_function_name):
    loss_function_name = loss_function_name.lower()
    if loss_function_name == 'esr_dc_prefilter':
        return LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])
    elif loss_function_name == 'esrloss':
        return ESRLoss()
    elif loss_function_name == 'dc_log_spectral_distance':
        return lambda output, target: 0.5 * DCLoss()(output, target) + 0.5 * log_spectral_distance(output, target)
    elif loss_function_name in globals():
        return globals()[loss_function_name]
    else:
        raise RuntimeError('Invalid loss function name.')

