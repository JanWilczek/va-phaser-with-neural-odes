import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
from common import plot_stft
from DigitalPhaser import DigitalPhaser, Oscillator


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('filepath', help='File to process with the phaser.')
    ap.add_argument('--lfo_frequency', default=0.34, type=float, help='Frequency in Hz of the low frequency oscillator controlling the allpass filters.')
    ap.add_argument('--allpass_modulation_index', default=0.2, type=float, help='Modulation index of the allpass filters'' cutoff frequencies.')
    return ap

def check_args(args):
    path = Path(args.filepath)
    if not path.exists():
        raise FileNotFoundError(f'Specified file path {path} does not exist.')

def phase_file(filepath: str, lfo_frequency, allpass_modulation_index, *args, **kwargs):
    phaser_input, fs = sf.read(filepath)

    # Take just the first channel of multichannel audio
    if phaser_input.ndim > 1:
        phaser_input = phaser_input[:, 0]
    
    output_directory = Path('phaser', 'models', 'digital_phaser', 'output')
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = Path(filepath).name[:-4]

    plot_stft(phaser_input, str((output_directory / filename)), sampling_rate=fs)
    
    base_frequencies = [500, 2000, 5000, 8000, 11000]
    allpass_cutoff_frequencies = [f/fs for f in base_frequencies + base_frequencies]
    phaser = DigitalPhaser(Oscillator(lfo_frequency, fs, lambda x: np.abs(np.sin(x/2))), allpass_cutoff_frequencies, allpass_modulation_index=allpass_modulation_index)
    phaser_output = phaser.process(phaser_input)

    output_filepath = str(output_directory / f'{filename}_phasered')
    sf.write(output_filepath + '.wav', phaser_output, fs)
    plot_stft(phaser_output, output_filepath, sampling_rate=fs)


def main():
    args = argument_parser().parse_args()
    check_args(args)
    phase_file(**vars(args))

if __name__ == '__main__':
    main()
