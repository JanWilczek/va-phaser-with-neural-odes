import argparse
import numpy as np
import json
from pathlib import Path
from common import normalized_mean_squared_error, signal_to_distortion_ratio, to_db
from presenters import get_signals


class MeasureExporter:
    def __init__(self):
        self.labels = []
        self.data = []

    def set_labels(self, labels: list):
        self.labels = labels

    def append(self, data):
        if len(data) != len(self.labels):
            raise RuntimeError('Passed-in data have a different dimension than the labels.')
        self.data.append(data)

    def export(self, filename):
        pass


class CSVExporter(MeasureExporter):
    def export(self, filename):
        header = ','.join(self.labels)
        np.savetxt(filename, self.data, delimiter=',', header=header)


def get_output_path(directory_path):
    return directory_path / 'test_output.wav'


def get_target_path(directory_path):
    args_path = directory_path / 'args.json'
    with open(args_path, 'r') as fp:
        args = json.load(fp)
    test_sampling_rate = args['test_sampling_rate']
    dataset_name = args['dataset_name']

    if dataset_name == 'diodeclip':
        prefix = 'diode_clipper'
    elif dataset_name == 'diode2clip':
        prefix = 'diode2_clipper'
    else:
        raise RuntimeError('invalid dataset name')

    if test_sampling_rate == 44100:
        sampling_rate_infix = ''
    else:
        sampling_rate_infix = f'{test_sampling_rate}Hz'

    target_filename = dataset_name + sampling_rate_infix + '-target.wav'

    target_path = Path(prefix) / 'data' / 'test' / target_filename

    return target_path


def get_measures(directory_path):
    target_path = get_target_path(directory_path)
    output_path = get_output_path(directory_path)

    args = object()
    args.estimated_signal_path = output_path
    args.clean_signal_path = target_path

    fs, target, output = get_signals(args)

    assert target.ndim == 1
    assert output.ndim == 1
    assert target.shape == output.shape

    nmse = to_db(normalized_mean_squared_error(output, target))
    sdr = signal_to_distortion_ratio(output, target)

    return nmse, sdr


if __name__=='__main__':
    # for each directory in the given ones
    parser = argparse.ArgumentParser(description='Compute NMSE and SDR of given models'''
                                                 ' outputs and store in a .csv file and in '
                                                 'a latex table.')
    parser.add_argument('run_directories', type=list,
                        help='a list of space-delimited paths to '
                             'directories containing the trained models.')
    args = parser.parse_args()

    exporter = CSVExporter()
    exporter.set_labels(['Model path','Normalized mean squared error [dB]', 'Signal-to-distortion ratio [dB]'])

    # compute NMSE and SDR between the test_output and the target
    nmse, sdr = get_measures(directory_path)

    # store in a .csv file
    exporter.append([directory_name, nmse, sdr])

    # and to a latex table


    exporter.export('nmse_sdr.csv')
