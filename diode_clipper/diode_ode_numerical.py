#!/usr/bin/python3
"""
Sample calls:
$ python -O diode_clipper/diode_ode_numerical.py -m forward_euler -u 38 -l 1 -s 5 -i 0 -f 128
$ python -O diode_clipper/diode_ode_numerical.py --method-name BDF --upsample-factor 8 --input-scaling-factor 20 --frame-length 0 --normalize
"""
import time
import argparse
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import resample, decimate
import torch
import torchaudio
import json
from tqdm import trange
from torchdiffeq import odeint
from CoreAudioML.training import ESRLoss
from NetworkTraining import create_dataset, get_run_name, save_json
from models.solvers import trapezoid_rule, forward_euler


SOLVERS = {'trapezoid_rule': trapezoid_rule,
           'forward_euler': forward_euler}

ODENET_SOLVERS = ['implicit_adams']


def argument_parser():
    parser = argparse.ArgumentParser(description='Run diode clipper ODE numerical solver.')
    parser.add_argument('-m', '--method-name', dest='method_name', default='forward_euler')
    parser.add_argument('-u', '--upsample-factor', dest='upsample_factor', default=1, type=int)
    parser.add_argument('-l', '--length-seconds', dest='length_seconds', default=0.0, type=float)
    parser.add_argument('-s', '--input-scaling-factor', dest='input_scaling_factor', default=1.0, type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-f', '--frame-length', dest='frame_length', default=22050, type=int)
    parser.add_argument('-p', '--plot', action='store_true')
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
        self.frame_length = None   # placeholder
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
        elif self.method_name in ODENET_SOLVERS:
            self.__method = lambda rhs, y0, t, args: odeint(
                partial(rhs, v_in=args[0], p=args[1], d=args[2]), y0, t, method=self.method_name)
            self.__rhs = diode_equation_rhs_torch
        else:
            self.__method = lambda rhs, y0, t, args: solve_ivp(
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

def scale_for_optimal_esr_and_clip(estimated_signal, true_signal):
    normalizing_factor = torch.sum(torch.multiply(estimated_signal, true_signal)) / torch.sum(torch.square(estimated_signal))
    estimated_signal *= normalizing_factor
    estimated_signal = torch.clip(estimated_signal, -1.0, 1.0)
    return estimated_signal

def run_solver(v_in, p):
    pre_samples = 10
    post_samples = 10
    v_in = torch.cat((torch.zeros((v_in.shape[0], 1, v_in.shape[2])),
                      v_in,
                      torch.zeros((v_in.shape[0], 1, v_in.shape[2]))),
                     axis=1)

    calculate_length = pre_samples + p.frame_length + post_samples
    resampled_t = torch.arange(0, calculate_length / p.sampling_rate, 1 /
                               (p.sampling_rate * p.upsample_factor), dtype=torch.float64)
    assert resampled_t.shape[0] == calculate_length * p.upsample_factor

    v_out = torch.zeros((p.frame_length, p.segments_count, 1))
    resampled_segment_length = p.upsample_factor * calculate_length
    solver_args = [None, p, DiodeParameters()]
    initial_value = torch.zeros((1, ), device=v_out.device)
    for segment_id in trange(1, p.segments_count + 1):  # Account for the zero-filled frame before the signal frames
        # Take pre_samples + p.frame_length + post_samples from the data to avoid resampling artifacts.
        # Additionally, we need to calculate more samples than the frame_length to get the next initial value.
        segment_data = torch.cat((v_in[-pre_samples:, segment_id - 1, 0],
                                  v_in[:, segment_id, 0],
                                  v_in[:post_samples, segment_id + 1, 0]),
                                 axis=0)

        scaled_segment_data = segment_data * p.input_scaling_factor

        assert scaled_segment_data.shape[0] == calculate_length

        resampled_scaled_segment_data = resample(scaled_segment_data, resampled_segment_length)

        assert resampled_scaled_segment_data.shape[0] / scaled_segment_data.shape[0] == p.upsample_factor

        solver_args[0] = resampled_scaled_segment_data

        y_segment_upsampled = p.method(
            p.rhs, initial_value, resampled_t, args=solver_args)

        assert y_segment_upsampled.shape[0] == calculate_length * p.upsample_factor

        v_out_segment = torch.from_numpy(resample(y_segment_upsampled, calculate_length))

        assert v_out_segment.shape[0] == calculate_length

        # Extract the samples corresponding to the current frame
        v_out[:, segment_id - 1, :] = v_out_segment[pre_samples:pre_samples + p.frame_length, :]

        # Initial value is the first valid sample output of the next processed "extended" frame.
        # Since from the current frame, the last post_samples will be taken, the initial value is
        # the first of these last post_samples samples.
        initial_value = v_out[- post_samples, segment_id - 1, :]

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
    p.frame_length = v_in.shape[0]

    # Check if the whole signal is to be analyzed
    if p.length_seconds <= 0.0:
        input_length_samples = v_in.shape[0] * v_in.shape[1]
        p.length_seconds = input_length_samples / p.sampling_rate

    start_time = time.time()
    print(f'Start time: {time.strftime("%H:%M:%S")}.')

    # Actual solver run, block by block
    v_out = run_solver(v_in, p)

    end_time = time.time()
    duration = end_time - start_time
    print(f'Finished in time {duration:.1f} seconds.')

    # Trim the true output to match the calculated one
    true_v_out_trimmed = true_v_out[:v_out.shape[0]]

    # Normalization
    if args.normalize:
        v_out = scale_for_optimal_esr_and_clip(v_out, true_v_out_trimmed)

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
    if args.plot:
        plt.figure()
        t = torch.arange(0, true_v_out_trimmed.shape[0] / p.sampling_rate, 1 / p.sampling_rate)
        plt.plot(t, true_v_out_trimmed, t, v_out)
        plt.legend(['ground truth', p.method_name])
        plt.savefig(p.plot_output_path)


if __name__ == '__main__':
    main()
