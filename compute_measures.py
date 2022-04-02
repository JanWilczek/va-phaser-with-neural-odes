"""Compute segmental SNR and frequency-weighted segmental SNR"""
"""python compute_measures.py -c diode_clipper/data/test/diodeclip-target.wav -e \
diode_clipper/runs/diodeclip/forward_euler/August12_08-14-18_axel_Hidden9Test44100/test_output.wav"""
import argparse
import subprocess
import re
from pathlib import Path
import numpy as np
import torch
from CoreAudioML import training
import pysepm
from presenters import append_int16, get_signals, argument_parser


def get_losses(clean_signal, estimated_signal):
    clean_signal_tensor = torch.Tensor(clean_signal)
    estimated_signal_tensor = torch.Tensor(estimated_signal)
    esr = training.ESRLoss()
    esr_value = esr(estimated_signal_tensor, clean_signal_tensor).item()
    loss_full = training.LossWrapper({'ESR': .5, 'DC': .5}, pre_filt=[1, -0.85])
    loss_full_value = loss_full(estimated_signal_tensor, clean_signal_tensor).item()
    return esr_value, loss_full_value


def get_peaq_measures(args):
    estimated_signal_int16_path = append_int16(args.estimated_signal_path)
    clean_signal_int16_path = append_int16(args.clean_signal_path)

    clean_signal_path = clean_signal_int16_path if clean_signal_int16_path.exists() else Path(args.clean_signal_path)
    estimated_signal_path = estimated_signal_int16_path if estimated_signal_int16_path.exists() else Path(args.estimated_signal_path)

    bash_command = f"./peaqb-fast/src/peaqb -r {str(clean_signal_path.resolve())} -t {str(estimated_signal_path.resolve())}".split()
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    di_regex = re.compile("DI: ([-+]?[0-9]*\.?[0-9]+)")
    odg_regex = re.compile("ODG: ([-+]?[0-9]*\.?[0-9]+)")

    di = float('nan')
    odg = float('nan')

    PEAQ_OUTPUT_FILENAME = 'analized'

    with open(PEAQ_OUTPUT_FILENAME, 'r') as f:
        lines = f.readlines()

        for line in lines:
            di_match = di_regex.match(line)
            if di_match:
                di = float(di_match.group(1))

            odg_match = odg_regex.match(line)
            if odg_match:
                odg = float(odg_match.group(1))

    peaq = odg + 5
    
    # Clean up
    clean_signal_int16_path.unlink(missing_ok=True)
    estimated_signal_int16_path.unlink(missing_ok=True)
    Path(PEAQ_OUTPUT_FILENAME).unlink()

    return di, odg, peaq


def main():
    args = argument_parser().parse_args()

    fs, clean_signal, estimated_signal = get_signals(args)
    
    # segSNR, fw-segSNR
    seg_snr = pysepm.SNRseg(clean_signal.clip(-1, 1), estimated_signal.clip(-1, 1), fs)
    fw_seg_snr = pysepm.fwSNRseg(clean_signal.clip(-1, 1), estimated_signal.clip(-1, 1), fs)

    # ESR
    esr_value, full_loss_value = get_losses(clean_signal, estimated_signal)
    
    # DI, ODG, PEAQ
    di, odg, peaq = get_peaq_measures(args)

    estimated_signal_dir = Path(args.estimated_signal_path).parent
    measures_output_file_path = estimated_signal_dir / 'measures.csv'

    measures = np.asarray([[seg_snr, fw_seg_snr, esr_value, full_loss_value, di, odg, peaq]])
    np.savetxt(measures_output_file_path, measures, delimiter=',', header='segSNR,fw-segSNR,ESR,ESR+DC+prefilter,DI,ODG,PEAQ')


if __name__ == '__main__':
    main()
