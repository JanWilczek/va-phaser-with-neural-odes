#!/usr/bin/python3
""" Sample call:
    python diode_clipper\diode_ode_numerical.py -m forward_euler -u 38 -l 1 -s 5 -i 0 -f 128
"""
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import resample
import torch
import torchaudio
import json
from CoreAudioML.training import ESRLoss
from NetworkTraining import create_dataset, get_run_name
from models.solvers import trapezoid_rule, forward_euler


SOLVERS = {'trapezoid_rule': trapezoid_rule, 
           'forward_euler': forward_euler}

def argument_parser():
    parser = argparse.ArgumentParser(description='Run diode clipper ODE numerical solver.')
    parser.add_argument('-m', '--method-name', dest='method_name', default='forward_euler')
    parser.add_argument('-u', '--upsample-factor', dest='upsample_factor', default=1, type=int)
    parser.add_argument('-l', '--length-seconds', dest='length_seconds', default=0.0, type=float)
    parser.add_argument('-s', '--input-scaling-factor', dest='input_scaling_factor', default=1.0, type=float)
    parser.add_argument('-i', '--interpolation-order', dest='interpolation_order', default=0, type=int)
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
        self.length_seconds = args.length_seconds # the same as input if <= 0 or unspecified
        self.interpolation_order = args.interpolation_order
        self.frame_length = args.frame_length
        
        self.save_json(vars(args), 'args.json')
                
        self.sampling_rate = None # placeholder

    def save_json(self, json_data, filename):
        with open(self.run_directory / filename, 'w') as f:
            json.dump(json_data, f, indent=4)

    @property
    def method_name(self):
        return self.__method_name

    @method_name.setter
    def method_name(self, method_name):
        self.__method_name = method_name
        self.run_directory = Path('diode_clipper', 'runs', 'ode_solver', self.method_name, get_run_name())
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.test_output_path = (self.run_directory / 'test_output.wav').resolve()


def jac_diode_equation_rhs(t, v_out, v_in, p, d):
    jac = - d.c1 - d.jac_c3 * np.cosh(v_out / d.v_t)
    return jac[:, None] # Jacobian needs to be of 1x1 size

def diode_equation_rhs(t, v_out, p):
    return (p.v_in(t) - v_out) / (p.R * p.C) - 2 * p.i_s / p.C * torch.sinh(v_out / p.v_t)

def diode_equation_rhs_numpy(t, v_out, v_in, p, d):
    return (v_in[int(t * p.sampling_rate * p.upsample_factor)] - v_out) * d.c1 - d.c2 * np.sinh(v_out / d.v_t)

def diode_equation_rhs_torch(t, v_out, v_in, p, d):
    return (v_in[int(t * p.sampling_rate * p.upsample_factor)] - v_out) * d.c1 - d.c2 * torch.sinh(v_out / d.v_t)

def main():
    args = argument_parser().parse_args()
    p = SimulationParameters(args)

    dataset = create_dataset(test_frame_len=args.frame_length)
    p.sampling_rate =  dataset.subsets['test'].fs
    true_v_out = dataset.subsets['test'].data['target'][0].permute(1, 0, 2).flatten()
    # p.v_in = dataset.subsets['test'].data['input'][0].squeeze()
    input_shape = dataset.subsets['test'].data['input'][0].shape
    input_length_samples = input_shape[0] * input_shape[1]
    if p.length_seconds <= 0.0:
        p.length_seconds = input_length_samples / p.sampling_rate
    segments_count = int(np.ceil(p.length_seconds * p.sampling_rate / args.frame_length))
    # true_v_out_trimmed = true_v_out[:p.t.shape[0]].unsqueeze(1)

    # initial_value = true_v_out[0].squeeze(1)

    start_time = time.time()

    if p.method_name in SOLVERS.keys():
        method = SOLVERS[p.method_name]
        rhs = diode_equation_rhs_torch
    else:
        method = lambda eq, y0, t, args: solve_ivp(eq, (t[0], t[-1]), initial_value, method=p.method_name, t_eval=t, args=args, jac=jac_diode_equation_rhs).y.T
        rhs = diode_equation_rhs_numpy

    # t = torch.arange(0, args.frame_length / p.sampling_rate, 1 / p.sampling_rate)
    resampled_t = torch.arange(0, args.frame_length / p.sampling_rate, 1 / (p.sampling_rate * p.upsample_factor))
    # t_span = (resampled_t[0], resampled_t[-1])
    y = torch.zeros((args.frame_length, segments_count, 1))
    for segment_id in range(segments_count):
        segment_data = dataset.subsets['test'].data['input'][0][:, segment_id, 0]
        scaled_segment_data = segment_data * args.input_scaling_factor
        resampled_scaled_segment_data = resample(scaled_segment_data.detach().numpy(), p.upsample_factor * scaled_segment_data.shape[0])
        initial_value = y[-1, max(segment_id - 1, 0), :]
        y_segment_upsampled = method(rhs, initial_value, resampled_t, args=[resampled_scaled_segment_data, p, DiodeParameters()])
        y_segment = torch.Tensor(resample(y_segment_upsampled, args.frame_length))
        y[:, segment_id, :] = y_segment
    y = y.permute(1, 0, 2).flatten().unsqueeze(1)
    true_v_out_trimmed = true_v_out[:y.shape[0]].unsqueeze(1)
    # else:
    #     raise NotImplementedError()
    #     print(f'Defaulting to scipy.integrate.solve_ivp/{p.method_name}.')
    #     t_span = (resampled_t[0], resampled_t[-1])
    #     result = solve_ivp(diode_equation_rhs, t_span, initial_value, method=p.method_name, t_eval=p.resampled_t, args=[p], jac=jac_diode_equation_rhs)
    #     print(result.message)
    #     y_upsampled = result.y
    
    # y = torch.Tensor(resample(y_upsampled, y_upsampled.shape[0] // p.upsample_factor))

    end_time = time.time()
    duration = end_time - start_time
    print(f'Finished in time {duration:.1f} seconds.')

    # Normalization
    v_out_result = y / torch.amax(torch.abs(y)) * torch.amax(torch.abs(true_v_out_trimmed))

    # The saved data needs to be transposed, because on Windows the Soundfile backend needs 
    # it to be of channels x frames (samples) shape. Sox, which is the default backend
    # on Mac/Linux chooses by itself what is the samples dimension and what is the channel dimension.
    torchaudio.save(p.test_output_path, v_out_result.T, p.sampling_rate)

    loss = ESRLoss()
    loss_result = loss(v_out_result, true_v_out_trimmed).item()

    print(f'ODESolver error: {loss_result}.')
    p.save_json({'time [s]': int(duration), 'ESRLoss': loss_result}, 'result.json')

    plt.figure()
    t = torch.arange(0, true_v_out_trimmed.shape[0] / p.sampling_rate, 1 / p.sampling_rate)
    plt.plot(t, true_v_out_trimmed.squeeze(), t, v_out_result.squeeze())
    plt.legend(['ground truth', p.method_name])
    plt.savefig((p.run_directory / f'diode_ode_{p.method_name}.png').resolve())

if __name__ == '__main__':
    main()
