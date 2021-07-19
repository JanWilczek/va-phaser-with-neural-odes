from pathlib import Path
from scipy.signal import resample
import soundfile as sf


def resample_file(filename, target_sampling_rate):
    hyphen_index = filename.index('-')
    resampled_filename = filename[:hyphen_index] + f'{target_sampling_rate}Hz' + filename[hyphen_index:]
    if not Path(resampled_filename).exists():
        print(f'Resampling {filename} to {target_sampling_rate} Hz...')
        data, sampling_rate = sf.read(filename)
        resampled_length = data.shape[0] * target_sampling_rate // sampling_rate
        resampled = resample(data, resampled_length, axis=0)
        sf.write(resampled_filename, resampled, target_sampling_rate)
    return resampled_filename

def resample_test_files(dataset_path, test_filename, target_sampling_rate):
    files = [dataset_path / (test_filename + '-input.wav'),
             dataset_path / (test_filename + '-target.wav'),]

    for file in files:
        resampled_filename = resample_file(str(file), target_sampling_rate)
    
    return resampled_filename[:str(resampled_filename).index('-')]
