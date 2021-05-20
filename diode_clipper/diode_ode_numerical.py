import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import torch
import torchaudio
from CoreAudioML.training import ESRLoss
from diode_benchmark import create_dataset, get_run_name


def jac_diode_equation_rhs(t, v_out, v_in, R, C, i_s, v_t):
    jac = - 1 / (R * C) - 2 * i_s / (C * v_t) * np.cosh(v_out / v_t)
    return jac[:, None] # Jacobian needs to be of 1x1 size

def diode_equation_rhs(t, v_out, v_in, R, C, i_s, v_t):
    # if int(t) % 1000 == 999:
        # print(t)
    return (v_in(t) - v_out) / (R * C) - 2 * i_s / C * np.sinh(v_out / v_t)

def main():
    method = 'BDF'
    run_directory = Path('diode_clipper', 'runs', 'ode_solver', method, get_run_name())
    run_directory.mkdir(parents=True, exist_ok=True)
    test_output_path = (run_directory / 'test_output.wav').resolve()

    dataset = create_dataset()
    VOLTAGE_SCALING_FACTOR = 5
    scaled_signal_in = dataset.subsets['test'].data['input'][0].squeeze() * VOLTAGE_SCALING_FACTOR
    true_v_out = dataset.subsets['test'].data['input'][0]
    t = np.arange(0, scaled_signal_in.shape[0])
    # seconds_length = 1.0
    # t = np.arange(0, int(seconds_length * dataset.subsets['test'].fs), dtype=int)
    t_span = (t[0], t[-1])
    initial_value = true_v_out[0].squeeze(1)
    R = 2.2e3
    C = 0.01e-6
    i_s = 5e-6
    v_t = 26e-3

    v_in = interp1d(t, scaled_signal_in[:(t[-1]+1)])

    start_time = time.time()
    result = solve_ivp(diode_equation_rhs, t_span, initial_value, method=method, t_eval=t, args=[v_in, R, C, i_s, v_t], jac=jac_diode_equation_rhs)
    end_time = time.time()

    print(result.message)
    print(f'Finished in time {end_time - start_time:.1f} seconds.')

    v_out_result = torch.Tensor(result.y / VOLTAGE_SCALING_FACTOR)
    torchaudio.save(test_output_path, v_out_result, dataset.subsets['test'].fs)

    loss = ESRLoss()
    loss_result = loss(v_out_result, true_v_out[:t[-1]+1]).item()

    print(f'ODESolver error: {loss_result}.')

if __name__ == '__main__':
    main()
