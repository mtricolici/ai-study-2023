import argparse
import vars

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="Video enhancer demo")
    parser.add_argument("-i", "--input-file", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output-file", required=True, help="Path to the output video file")
    parser.add_argument("-f", "--fps", type=int, default=None, help="Frames per second (default: it will be detected automatically)")
    parser.add_argument("-m", "--model_name", default="NAFNet-REDS-width64", help="Model name (default: NAFNet-REDS-width64)")
    args = parser.parse_args()
    vars.source_file = args.input_file
    vars.target_file = args.output_file
    vars.fps = args.fps
    vars.model_name = args.model_name
