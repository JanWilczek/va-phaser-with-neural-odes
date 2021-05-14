from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import torchaudio
from CoreAudioML.training import ESRLoss
from diode_benchmark import create_dataset


def diode_equation_rhs(t, v_out, v_in, R, C, i_s, v_t):
    (v_in[t] - v_out) / (R * C) - 2 * i_s / C * np.sinh(v_out / v_t)

def main():
    dataset = create_dataset()
    VOLTAGE_SCALING_FACTOR = 5
    v_in = dataset.subsets['test'].data['input'][0] * VOLTAGE_SCALING_FACTOR
    true_v_out = dataset.subsets['test'].data['input'][0]
    t = np.arange(0, v_in.shape[0])
    t_span = (t[0], t[-1])
    initial_value = true_v_out[0].item()
    R = 2.2e3
    C = 0.01e-6
    i_s = 5e-6
    v_t = 26e-3

    result = solve_ivp(diode_equation_rhs, t_span, initial_value, args=[v_in, R, C, i_s, v_t])
    v_out_result = result[1] / VOLTAGE_SCALING_FACTOR

    loss = ESRLoss()
    loss_result = loss(v_out_result, true_v_out).item()

    run_directory = Path('diode_clipper', 'runs', 'ode_solver', 'RK45')
    test_output_path = (run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, v_out_result, dataset.subsets['test'].fs)
    print(f'ODESolver error: {loss_result}.')

if __name__ == '__main__':
    main()
