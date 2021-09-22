import numpy as np
from scipy.io import wavfile


def convert_audio_data_float32_to_int16(audio_data):
    assert audio_data.dtype == np.float32, 'Data to be converted is not in float32 format.'
    return (audio_data.clip(-1, 1) * 32767).astype(np.int16, order='C')

def convert_audio_file_float32_to_int16(source_filepath, destination_filepath):
    fs, data = wavfile.read(source_filepath)
    data = convert_audio_data_float32_to_int16(data)
    wavfile.write(destination_filepath, fs, data)
