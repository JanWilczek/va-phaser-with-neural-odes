import argparse
from common import convert_audio_file_float32_to_int16


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dest')
    return ap

def main():
    args = argument_parser().parse_args()
    convert_audio_file_float32_to_int16(args.src, args.dest)

if __name__ == '__main__':
    main()
