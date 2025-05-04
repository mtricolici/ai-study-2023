import onnxruntime as ort
ort.set_default_logger_severity(3)

import argparse

from cmdparser import parse_cmd_args
import vars
import ffmpeg
import convert

def main():
    parse_cmd_args()

    ffmpeg.detect_fps()
    ffmpeg.check_audio_stream_presence()
    if vars.has_audio:
      ffmpeg.save_audio_stream()

    ffmpeg.extract_frames()
    convert.process_frames()

    ffmpeg.create_output_video()

if __name__ == "__main__":
    main()
