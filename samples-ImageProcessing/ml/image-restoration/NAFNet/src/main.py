import argparse

from cmdparser import parse_cmd_args
import vars
import ffmpeg
import convert

def main():
    parse_cmd_args()
    ffmpeg.detect_fps()
    ffmpeg.extract_frames()
    convert.process_frames()
    ffmpeg.create_output_video()

if __name__ == "__main__":
    main()
