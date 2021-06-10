#!/usr/bin/python3
"""
Sample call:
$ python diode_clipper\diode_ode_numerical.py -m forward_euler -u 38 -l 1 -s 5 -i 0 -f 128
"""
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import resample
import torch
import torchaudio
import json
from tqdm import trange
from CoreAudioML.training import ESRLoss
from NetworkTraining import create_dataset, get_run_name, save_json
from models.solvers import trapezoid_rule, forward_euler


SOLVERS = {'trapezoid_rule': trapezoid_rule,
           'forward_euler': forward_euler}


def argument_parser():
    parser = argparse.ArgumentParser(description='Run diode clipper ODE numerical solver.')
    parser.add_argument('-m', '--method-name', dest='method_name', default='forward_euler')
    parser.add_argument('-u', '--upsample-factor', dest='upsample_factor', default=1, type=int)
    parser.add_argument('-l', '--length-seconds', dest='length_seconds', default=0.0, type=float)
    parser.add_argument('-s', '--input-scaling-factor', dest='input_scaling_factor', default=1.0, type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-f', '--frame-length', dest='frame_length', default=22050, type=int)
    return parser


class DiodeParameters:
    def __init__(self):
        # From "Numerical Methods for Simulation of Guitar Distortion Circuits" by Yeh et al.
        self.R = 2.2e3
        self.C = 10e-9
        self.i_s = 2.52e-9
        self.v_t = 45.3e-3

        self.c1 = 1 / (self.R * self.C)
        self.c2 = 2 * self.i_s / self.C
        self.jac_c3 = 2 * self.i_s / (self.C * self.v_t)


class SimulationParameters:
    def __init__(self, args):
        self.method_name = args.method_name
        self.upsample_factor = args.upsample_factor
        self.input_scaling_factor = args.input_scaling_factor
        self.length_seconds = args.length_seconds  # the same as input if <= 0 or unspecified
        self.frame_length = args.frame_length
        self.sampling_rate = None  # placeholder

    @property
    def method_name(self):
        return self.__method_name

    @method_name.setter
    def method_name(self, method_name):
        self.__method_name = method_name
        self.run_directory = Path('diode_clipper', 'runs', 'ode_solver', self.method_name, get_run_name())
        self.run_directory.mkdir(parents=True, exist_ok=False)
        if self.method_name in SOLVERS.keys():
            self.__method = SOLVERS[self.method_name]
            self.__rhs = diode_equation_rhs_torch
        else:
            self.__method =  lambda rhs, y0, t, args: solve_ivp(
                rhs, (t[0], t[-1]), y0, method=self.method_name, t_eval=t, args=args, jac=jac_diode_equation_rhs).y.T
            self.__rhs = diode_equation_rhs_numpy

    @property
    def test_output_path(self):
        return (self.run_directory / 'test_output.wav').resolve()

    @property
    def plot_output_path(self):
        return (self.run_directory / f'diode_ode_{self.method_name}.png').resolve()

    @property
    def args_output_path(self):
        return self.run_directory / 'args.json'

    @property
    def results_output_path(self):
        return self.run_directory / 'result.json'

    @property
    def segments_count(self):
        if self.length_seconds <= 0.0:
            raise ValueError("length_seconds was not set!")
        return int(np.ceil(self.length_seconds * self.sampling_rate / self.frame_length))

    @property
    def method(self):
        return self.__method

    @property
    def rhs(self):
        return self.__rhs


def jac_diode_equation_rhs(t, v_out, v_in, p, d):
    jac = - d.c1 - d.jac_c3 * np.cosh(v_out / d.v_t)
    return jac[:, None]  # Jacobian needs to be of 1x1 size


def diode_equation_rhs_numpy(t, v_out, v_in, p, d):
    return (v_in[int(t * p.sampling_rate * p.upsample_factor)] - v_out) * d.c1 - d.c2 * np.sinh(v_out / d.v_t)


def diode_equation_rhs_torch(t, v_out, v_in, p, d):
    return (v_in[int(t * p.sampling_rate * p.upsample_factor)] - v_out) * d.c1 - d.c2 * torch.sinh(v_out / d.v_t)


def run_solver(v_in, p):
    v_in = torch.cat((v_in, torch.zeros((v_in.shape[0], 1, v_in.shape[2]))), axis=1)

    # We need 1 additional sample for the initial value of the next frame
    calculate_length = p.frame_length + 1
    resampled_t = torch.arange(0, calculate_length / p.sampling_rate, 1 / (p.sampling_rate * p.upsample_factor))
    assert resampled_t.shape[0] == calculate_length * p.upsample_factor
    v_out = torch.zeros((p.frame_length, p.segments_count, 1))
    resampled_segment_length = p.upsample_factor * calculate_length
    solver_args = [None, p, DiodeParameters()]
    initial_value = torch.zeros((1, ), device=v_out.device)
    for segment_id in trange(p.segments_count):
        segment_data = torch.cat((v_in[:, segment_id, 0], v_in[0, segment_id + 1, :]), axis=0)
        scaled_segment_data = segment_data * p.input_scaling_factor
        resampled_scaled_segment_data = resample(scaled_segment_data, resampled_segment_length)
        solver_args[0] = resampled_scaled_segment_data

        y_segment_upsampled = p.method(
            p.rhs, initial_value, resampled_t, args=solver_args)

        v_out_segment = torch.Tensor(resample(y_segment_upsampled, calculate_length))
        v_out[:, segment_id, :] = v_out_segment[:p.frame_length]

        # The last sample of the last output is the first sample of the next output
        initial_value = v_out_segment[-1, :]

    v_out = v_out.permute(1, 0, 2).flatten()
    return v_out


def main():
    # Process input arguments
    args = argument_parser().parse_args()
    p = SimulationParameters(args)
    save_json(vars(args), p.args_output_path)

    dataset = create_dataset(test_frame_len=args.frame_length)
    p.sampling_rate = dataset.subsets['test'].fs
    true_v_out = dataset.subsets['test'].data['target'][0].permute(1, 0, 2).flatten()
    v_in = dataset.subsets['test'].data['input'][0]

    # Check if the whole signal is to be analyzed
    if p.length_seconds <= 0.0:
        input_length_samples = v_in.shape[0] * v_in.shape[1]
        p.length_seconds = input_length_samples / p.sampling_rate

    start_time = time.time()

    # Actual solver run, block by block
    v_out = run_solver(v_in, p)

    end_time = time.time()
    duration = end_time - start_time
    print(f'Finished in time {duration:.1f} seconds.')

    # Trim the true output to match the calculated one
    true_v_out_trimmed = true_v_out[:v_out.shape[0]]

    # Normalization
    if args.normalize:
        v_out = v_out / torch.amax(torch.abs(v_out)) * torch.amax(torch.abs(true_v_out_trimmed))

    # Store the audio output
    # The saved data needs to be transposed, because on Windows the Soundfile backend needs
    # it to be of channels x frames (samples) shape. Sox, which is the default backend
    # on Mac/Linux chooses by itself what is the samples dimension and what is the channel dimension.
    torchaudio.save(p.test_output_path, v_out.unsqueeze(0), p.sampling_rate)

    # Calculate loss
    loss = ESRLoss()
    loss_result = loss(v_out, true_v_out_trimmed).item()

    print(f'ODESolver error: {loss_result}.')
    save_json({'time [s]': int(duration), 'ESRLoss': loss_result}, p.results_output_path)

    # Plot result
    plt.figure()
    t = torch.arange(0, true_v_out_trimmed.shape[0] / p.sampling_rate, 1 / p.sampling_rate)
    plt.plot(t, true_v_out_trimmed, t, v_out)
    plt.legend(['ground truth', p.method_name])
    plt.savefig(p.plot_output_path)


if __name__ == '__main__':
    main()
