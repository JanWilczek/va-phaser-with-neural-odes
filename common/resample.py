import os
from pathlib import Path
from scipy.signal import resample
import soundfile as sf


def resample_file_if_not_already_resampled(filepath: Path, target_sampling_rate: int):
    hyphen_index = filepath.name.index('-')
    resampled_filename = filepath.name[:hyphen_index] + f'{target_sampling_rate}Hz' + filepath.name[hyphen_index:]
    resampled_filepath = filepath.parent / resampled_filename
    if not resampled_filepath.exists():
        print(f'Resampling {filepath} to {target_sampling_rate} Hz and storing in {resampled_filepath}...')
        resample_file(filepath, resampled_filepath, target_sampling_rate)
    return resampled_filename

def resample_file(filepath: Path, resampled_filepath: Path, target_sampling_rate: int):
    data, sampling_rate = sf.read(filepath)
    resampled_length = data.shape[0] * target_sampling_rate // sampling_rate
    resampled = resample(data, resampled_length, axis=0)
    sf.write(resampled_filepath, resampled, target_sampling_rate)

def resample_test_files(dataset_path: Path, test_filename: str, target_sampling_rate: int):
    files = [dataset_path / (test_filename + '-input.wav'),
             dataset_path / (test_filename + '-target.wav'),]

    for file in files:
        resampled_filename = resample_file_if_not_already_resampled(file, target_sampling_rate)
    
    # Prepend additional directories, if they were present in test_filename.
    return os.path.join(test_filename[:test_filename.rindex(os.path.sep)], resampled_filename[:resampled_filename.index('-')])
