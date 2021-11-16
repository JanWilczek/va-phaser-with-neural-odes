from pathlib import Path
from argparse import ArgumentParser
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz


def plot_time_alignment(target, ours, theirs, output_filepath, sampling_rate):
    setup_pyplot_for_latex()

    t = np.arange(0, target.shape[0]) / sampling_rate

    plt.figure()
    plt.plot(t, target)
    plt.plot(t, ours, '--')
    plt.plot(t, theirs, ':')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(['Target', 'ODENet30(FE)', 'LSTM16'])
    plt.xlim([0, t[-1]])
    plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=600, transparent=True)
    save_tikz(output_filepath)

def preprocess_audio_file(filepath, seconds):
    filepath = Path(filepath)
    data, fs = sf.read(filepath, always_2d=True)
    data = data[:, 0] # Take just the first channel of audio
    if seconds:
        data = data[:int(fs * seconds)]
    return data, fs

def main():
    ap = ArgumentParser()
    ap.add_argument('--target', help='.wav file with a fragment of the target signal.')
    ap.add_argument('--ours', help='.wav file with a fragment of the ours signal with our method.')
    ap.add_argument('--theirs', help='.wav file with a fragment of the ours signal with their method.')
    ap.add_argument('--seconds', default=None, type=float, help='number of seconds into the file to take.')
    args = ap.parse_args()
    
    target, fs = preprocess_audio_file(args.target, args.seconds)
    ours, fs_ours = preprocess_audio_file(args.ours, args.seconds)
    theirs, fs_theirs = preprocess_audio_file(args.theirs, args.seconds)
    
    assert fs == fs_ours
    assert fs_theirs == fs_ours
    
    output_filename = 'time_alignment_comparison'
    output_path = Path(args.target).parent / output_filename
    plot_time_alignment(target, ours, theirs, output_path, fs)

if __name__=='__main__':
    main()