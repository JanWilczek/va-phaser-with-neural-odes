import argparse
import numpy as np
import json
from pathlib import Path
import torch
from common import normalized_mean_squared_error, signal_to_distortion_ratio, to_db
from presenters import get_signals


class Object(object):
    pass


def root_repository_dir():
    return Path().absolute().parent


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
    def transform(self):
        header = ','.join(self.labels)
        lines = ['\n' + ','.join([str(d) for d in line]) for line in self.data]
        lines.insert(0, header)
        return ''.join(lines)

    def export(self, filename):
        filename = filename.with_suffix('.csv')
        content = self.transform()
        with open(filename, 'w') as f:
            f.write(content)


class LatexTableExporter(MeasureExporter):
    def transform(self):
        table_lines = []
        table_lines.append('\\begin{tabular}')
        table_lines.append(r'\toprule')
        table_lines.append(' & '.join([r'\textbf{' + label + r'}' for label in self.labels]))
        table_lines.append(r'\\ \midrule')
        for data in self.data:
            table_lines.append(' & '.join([str(d) for d in data]))
            table_lines.append(r'\\')
        table_lines.append(r'\bottomrule')
        table_lines.append('\\end{tabular}')
        return '\n'.join(table_lines)

    def export(self, filename: Path):
        filename = filename.with_suffix('.tex')
        content = self.transform()
        with open(filename, 'w') as f:
            f.write(content)


class CompositeExporter(MeasureExporter):
    def __init__(self):
        super().__init__()
        self.exporters = []

    def add_exporter(self, exporter):
        self.exporters.append(exporter)

    def set_labels(self, labels: list):
        for exporter in self.exporters:
            exporter.set_labels(labels)

    def append(self, data):
        for exporter in self.exporters:
            exporter.append(data)

    def export(self, filename):
        for exporter in self.exporters:
            exporter.export(filename)


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

    target_path = root_repository_dir() / prefix / 'data' / 'test' / target_filename

    return target_path


def get_measures(directory_path: Path):
    target_path = get_target_path(directory_path)
    output_path = get_output_path(directory_path)

    args = Object()
    args.estimated_signal_path = output_path
    args.clean_signal_path = target_path

    fs, target, output = get_signals(args)

    if target.ndim > 1:
        target = target[..., 0]
    if output.ndim > 1:
        output = output[..., 0]

    assert target.ndim == 1
    assert output.ndim == 1
    assert target.shape == output.shape

    target = torch.tensor(target)
    output = torch.tensor(output)

    nmse_db = to_db(normalized_mean_squared_error(output, target))
    sdr = signal_to_distortion_ratio(output, target)

    return nmse_db.item(), sdr.item()


if __name__=='__main__':
    # for each directory in the given ones
    parser = argparse.ArgumentParser(description='Compute NMSE and SDR of given models'''
                                                 ' outputs and store in a .csv file and in '
                                                 'a latex table.')
    parser.add_argument('run_directories', default=[], nargs='+',
                        help='a list of space-delimited paths to '
                             'directories containing the trained models.')
    args = parser.parse_args()

    exporter = CompositeExporter()
    exporter.add_exporter(CSVExporter())
    exporter.add_exporter(LatexTableExporter())
    exporter.set_labels(['Model path', 'Normalized mean squared error [dB]',
                         'Signal-to-distortion ratio [dB]'])

    for directory_name in args.run_directories:
        directory_path = root_repository_dir() / Path(directory_name)
        # compute NMSE and SDR between the test_output and the target
        nmse_db, sdr = get_measures(directory_path)

        exporter.append([directory_name, nmse_db, sdr])

    exporter.export(root_repository_dir() / 'nmse_sdr')
