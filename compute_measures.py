"""Compute segmental SNR and frequency-weighted segmental SNR"""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import pysepm


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean_signal_path', '-c', required=True)
    ap.add_argument('--estimated_signal_path', '-e', required=True)
    return ap


def main():
    args = argument_parser().parse_args()
    fs_clean, clean_signal = wavfile.read(args.clean_signal_path)
    fs, estimated_signal = wavfile.read(args.estimated_signal_path)

    assert fs == fs_clean, 'Clean and estimated signal must have the same sampling rate.'

    seg_snr = pysepm.SNRseg(clean_signal, estimated_signal, fs)
    fw_seg_snr = pysepm.fwSNRseg(clean_signal, estimated_signal, fs)

    estimated_signal_dir = Path(args.estimated_signal_path).parent
    measures_output_file_path = estimated_signal_dir / 'measures.csv'

    measures = np.asarray([[seg_snr, fw_seg_snr]])
    np.savetxt(measures_output_file_path, measures, delimiter=',', header='segSNR; fw-segSNR')


if __name__ == '__main__':
    main()
