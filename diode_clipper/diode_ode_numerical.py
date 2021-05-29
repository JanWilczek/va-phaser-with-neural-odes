import time
import argparse
from pathlib import Path
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import resample
import torch
import torchaudio
from CoreAudioML.training import ESRLoss
from NetworkTraining import create_dataset, get_run_name
from models.solvers import trapezoid_rule, forward_euler


parser = argparse.ArgumentParser(description='Run diode clipper ODE numerical solver.')
parser.add_argument('-m', '--method-name', dest='method_name', default='forward_euler')
parser.add_argument('-u', '--upsample-factor', dest='upsample_factor', default=1, type=int)
parser.add_argument('-l', '--length-seconds', dest='length_seconds', default=0.0, type=float)
parser.add_argument('-s', '--input-scaling-factor', dest='input_scaling_factor', default=1.0, type=float)

SOLVERS = {'trapezoid_rule': trapezoid_rule, 
           'forward_euler': forward_euler}

class SimulationParameters:
    def __init__(self, args):
        self.method_name = args.method_name
        self.upsample_factor = args.upsample_factor
        self.input_scaling_factor = args.input_scaling_factor
        self.length_seconds = args.length_seconds # the same as input if <= 0 or unspecified
        
        # From "Numerical Methods for Simulation of Guitar Distortion Circuits" by Yeh et al.
        self.R = 2.2e3
        self.C = 10e-9
        self.i_s = 2.52e-9
        self.v_t = 45.3e-3
        self.__v_in = None # placeholder
        self.sampling_rate = None # placeholder
        self.t = None # placeholder

    @property
    def method_name(self):
        return self.__method_name

    @method_name.setter
    def method_name(self, method_name):
        self.__method_name = method_name
        self.run_directory = Path('diode_clipper', 'runs', 'ode_solver', self.method_name, get_run_name())
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.test_output_path = (self.run_directory / 'test_output.wav').resolve()

    @property
    def v_in(self):
        return self.__v_in

    @v_in.setter
    def v_in(self, v_in):
        scaled_signal_in = v_in * self.input_scaling_factor
        if not hasattr(self, 'length_seconds') or self.length_seconds <= 0.0:
            self.length_seconds = scaled_signal_in.shape[0] / self.sampling_rate
        self.t = torch.arange(0, self.length_seconds, 1 / self.sampling_rate)

        trimmed_scaled_signal_in = scaled_signal_in[:self.t.shape[0]]
        resampled_scaled_signal_in, resampled_t = resample(trimmed_scaled_signal_in.detach().numpy(), self.upsample_factor * trimmed_scaled_signal_in.shape[0], self.t.detach().numpy())
        self.__v_in = resampled_scaled_signal_in
        self.resampled_t = resampled_t

def jac_diode_equation_rhs(t, v_out, p):
    jac = - 1 / (R * C) - 2 * i_s / (C * v_t) * torch.cosh(v_out / v_t)
    return jac[:, None] # Jacobian needs to be of 1x1 size

def diode_equation_rhs(t, v_out, p):
    v_in_value = p.v_in[int(t * p.sampling_rate * p.upsample_factor)]
    return (v_in_value - v_out) / (p.R * p.C) - 2 * p.i_s / p.C * torch.sinh(v_out / p.v_t)

def main():
    args = parser.parse_args()
    p = SimulationParameters(args)

    dataset = create_dataset()
    p.sampling_rate =  dataset.subsets['test'].fs
    true_v_out = dataset.subsets['test'].data['input'][0]
    p.v_in = dataset.subsets['test'].data['input'][0].squeeze()

    initial_value = true_v_out[0].squeeze(1)

    start_time = time.time()

    if p.method_name in SOLVERS.keys():
        method = SOLVERS[p.method_name]
        # v_in = interp1d(resampled_t, resampled_scaled_signal_in)
        y_upsampled = method(diode_equation_rhs, initial_value, torch.from_numpy(p.resampled_t), args=[p])
        y = resample(y_upsampled, y_upsampled.shape[0] // p.upsample_factor)
    else:
        print(f'Defaulting to scipy.integrate.solve_ivp/{p.method_name}.')
        t_span = (p.t[0], p.t[-1])
        # v_in = interp1d(t, trimmed_scaled_signal_in)
        result = solve_ivp(diode_equation_rhs, t_span, initial_value, method=p.method_name, t_eval=p.resampled_t, args=p, jac=jac_diode_equation_rhs)
        print(result.message)
        y = result.y
    
    end_time = time.time()

    print(f'Finished in time {end_time - start_time:.1f} seconds.')

    v_out_result = torch.Tensor(y)

    # The saved data needs to be transposed, because on Windows the Soundfile backend needs 
    # it to be of channels x frames (samples) shape. Sox, which is the default backend
    # on Mac/Linux chooses by itself what is the samples dimension and what is the channel dimension.
    torchaudio.save(p.test_output_path, v_out_result.T, p.sampling_rate)

    loss = ESRLoss()
    v_out_result_1d = v_out_result.squeeze()
    true_v_out_trimmed = true_v_out[:p.t.shape[0]].squeeze()
    loss_result = loss(v_out_result_1d, true_v_out_trimmed).item()

    print(f'ODESolver error: {loss_result}.')

    plt.figure()
    plt.plot(p.t, true_v_out_trimmed, p.t, v_out_result_1d)
    plt.legend(['ground truth', p.method_name])
    plt.savefig((p.run_directory / f'diode_ode_{p.method_name}.png').resolve())

if __name__ == '__main__':
    main()
