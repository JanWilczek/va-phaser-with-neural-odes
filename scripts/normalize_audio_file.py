import argparse
from pathlib import Path
import soundfile as sf
import pyloudnorm as pyln
import numpy as np


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('filepath', help='Path to the audio file')
    ap.add_argument('--target_loudness', '-lufs', default=-12, type=float, help='Loudness in LUFS to normalize to (default: %(default)s).')
    return ap


def normalize_audio_file(filepath: Path, target_loudness: int):
    PROCESSED_FILE_SUFFIX = '_normalized'
    
    if filepath.stem.endswith(PROCESSED_FILE_SUFFIX):
        return
    
    data, rate = sf.read(filepath)

    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_loudness)

    normalized_filepath = filepath.with_name(filepath.stem + PROCESSED_FILE_SUFFIX + filepath.suffix)

    if (np.any(np.abs(np.array(loudness_normalized_audio)) >= 1.0)):
        print(f"WARNING! Clipping detected in {normalized_filepath}.")

    sf.write(normalized_filepath, loudness_normalized_audio, rate)

    print(f'Normalized {filepath.name}. Original loudness: {loudness} LUFS. New loudness: {target_loudness} LUFS. Stored in {normalized_filepath}.')


def main():
    args = argument_parser().parse_args()
    normalize_audio_file(Path(args.filepath), args.target_loudness)


if __name__ == "__main__":
    main()
