from scipy.io import wavfile
import numpy as np
from pathlib import Path
from CoreAudioML import dataset
from common import convert_audio_file_float32_to_int16


def get_signals(args):
    fs, estimated_signal  = get_convert_signal(args.estimated_signal_path)
    fs_clean, clean_signal  = get_convert_signal(args.clean_signal_path)

    assert fs == fs_clean, 'Clean and estimated signal must have the same sampling rate.'

    # Trim signals to common length
    shorter_length = min(clean_signal.shape[0], estimated_signal.shape[0])
    clean_signal = clean_signal[:shorter_length]
    estimated_signal = estimated_signal[:shorter_length]

    return fs, clean_signal, estimated_signal


def get_convert_signal(filepath : str):
    fs, signal = wavfile.read(filepath)

    if signal.dtype == np.int16:
        signal = dataset.audio_converter(signal)
    else:
        convert_audio_file_float32_to_int16(filepath, append_int16(filepath))

    return fs, signal


def append_int16(filepath):
    filepath = Path(filepath)
    dir = filepath.parent
    filename = filepath.name[:-4] + '_int16.wav'
    return dir / filename
