"""Compute segmental SNR and frequency-weighted segmental SNR"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from CoreAudioML import training
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

    # Trim signals to common length
    shorter_length = min(clean_signal.shape[0], estimated_signal.shape[0])
    clean_signal = clean_signal[:shorter_length].astype(float)
    estimated_signal = estimated_signal[:shorter_length].astype(float)

    # segSNR, fw-segSNR
    seg_snr = pysepm.SNRseg(clean_signal, estimated_signal, fs)
    fw_seg_snr = pysepm.fwSNRseg(clean_signal, estimated_signal, fs)

    # ESR
    esr = training.ESRLoss()
    esr_value = esr(torch.from_numpy(estimated_signal), torch.from_numpy(clean_signal)).item()
    loss_full = training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])
    loss_full_value = loss_full(torch.from_numpy(estimated_signal), torch.from_numpy(clean_signal)).item()

    estimated_signal_dir = Path(args.estimated_signal_path).parent
    measures_output_file_path = estimated_signal_dir / 'measures.csv'

    measures = np.asarray([[seg_snr, fw_seg_snr, esr_value, loss_full_value]])
    np.savetxt(measures_output_file_path, measures, delimiter=',', header='segSNR,fw-segSNR,ESR,ESR+DC+prefilter')


if __name__ == '__main__':
    main()
