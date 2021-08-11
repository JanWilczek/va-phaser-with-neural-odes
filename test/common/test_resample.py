from pathlib import Path
from scipy.io import wavfile
import torchaudio
from common import resample_test_files


def print_stats(waveform, sample_rate=None, src=None):
    """Copied from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html"""
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")

def print_wavfile_info(path):
    # metadata = torchaudio.info(path)
    # print(metadata.__dict__)
    fs, waveform = wavfile.read(path)
    print_stats(waveform, fs, path)

def main():
    test_files_directory = Path('diode_clipper', 'data', 'test')
    test_file_name = 'diodeclip'
    target_sampling_rate = 22050
    resampled_file_prefix = resample_test_files(test_files_directory, test_file_name, target_sampling_rate)

    print_wavfile_info(test_files_directory / (test_file_name + '-input.wav'))
    print_wavfile_info(test_files_directory / (resampled_file_prefix + '-input.wav'))


if __name__ == '__main__':
    main()
