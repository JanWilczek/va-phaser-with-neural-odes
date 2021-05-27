import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import resample
import torch
import torchaudio
from CoreAudioML.training import ESRLoss
from diode_benchmark import create_dataset, get_run_name
from models.solvers import forward_euler


def jac_diode_equation_rhs(t, v_out, v_in, R, C, i_s, v_t):
    jac = - 1 / (R * C) - 2 * i_s / (C * v_t) * torch.cosh(v_out / v_t)
    return jac[:, None] # Jacobian needs to be of 1x1 size

def diode_equation_rhs(t, v_out, v_in, R, C, i_s, v_t):
    # if int(t) % 1000 == 999:
        # print(t)
    v_in_value = torch.from_numpy(v_in(t))
    return (v_in_value - v_out) / (R * C) - 2 * i_s / C * torch.sinh(v_out / v_t)

def main():
    method = 'forward_euler'
    run_directory = Path('diode_clipper', 'runs', 'ode_solver', method, get_run_name())
    run_directory.mkdir(parents=True, exist_ok=True)
    test_output_path = (run_directory / 'test_output.wav').resolve()

    RESAMPLE_FACTOR = 38
    VOLTAGE_SCALING_FACTOR = 5

    dataset = create_dataset()
    sampling_rate =  dataset.subsets['test'].fs
    true_v_out = dataset.subsets['test'].data['input'][0]
    scaled_signal_in = dataset.subsets['test'].data['input'][0].squeeze() * VOLTAGE_SCALING_FACTOR
    # seconds_length = 1
    seconds_length = scaled_signal_in.shape[0] / sampling_rate
    t = torch.arange(0, seconds_length, 1 / sampling_rate)
    t_span = (t[0], t[-1])
    initial_value = true_v_out[0].squeeze(1)
    trimmed_scaled_signal_in = scaled_signal_in[:t.shape[0]]
    resampled_scaled_signal_in, resampled_t = resample(trimmed_scaled_signal_in.detach().numpy(), RESAMPLE_FACTOR * trimmed_scaled_signal_in.shape[0], t.detach().numpy())

    # From "Numerical Methods for Simulation of Guitar Distortion Circuits" by Yeh et al.
    R = 2.2e3
    C = 10e-9
    i_s = 2.52e-9
    v_t = 45.3e-3

    start_time = time.time()

    if method == 'forward_euler':
        v_in = interp1d(resampled_t, resampled_scaled_signal_in)
        rhs_args = [v_in, R, C, i_s, v_t]
        y_upsampled = forward_euler(diode_equation_rhs, initial_value, torch.from_numpy(resampled_t), args=rhs_args)
        y = resample(y_upsampled, y_upsampled.shape[0] // RESAMPLE_FACTOR)
        assert y.shape[0] == t.shape[0]
    else:
        v_in = interp1d(t, trimmed_scaled_signal_in)
        rhs_args = [v_in, R, C, i_s, v_t]
        result = solve_ivp(diode_equation_rhs, t_span, initial_value, method=method, t_eval=t, args=rhs_args, jac=jac_diode_equation_rhs)
        print(result.message)
        y = result.y
    
    end_time = time.time()

    print(f'Finished in time {end_time - start_time:.1f} seconds.')

    v_out_result = torch.Tensor(y)

    # The saved data needs to be transposed, because on Windows the Soundfile backend needs 
    # it to be of channels x frames (samples) shape. Sox, which is the default backend
    # on Mac/Linux chooses by itself what is the samples dimension and what is the channel dimension.
    torchaudio.save(test_output_path, v_out_result.T, sampling_rate)

    loss = ESRLoss()
    v_out_result_1d = v_out_result.squeeze()
    true_v_out_trimmed = true_v_out[:t.shape[0]].squeeze()
    loss_result = loss(v_out_result_1d, true_v_out_trimmed).item()

    print(f'ODESolver error: {loss_result}.')

    plt.figure()
    plt.plot(t, true_v_out_trimmed, t, v_out_result_1d)
    plt.legend(['ground truth', method])
    plt.savefig((run_directory / f'diode_ode_{method}.png').resolve())

if __name__ == '__main__':
    main()
