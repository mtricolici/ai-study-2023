import argparse

from cmdparser import parse_cmd_args
import vars
import ffmpeg

def main():
    parse_cmd_args()
    ffmpeg.detect_fps()
    ffmpeg.extract_frames()

if __name__ == "__main__":
    main()
