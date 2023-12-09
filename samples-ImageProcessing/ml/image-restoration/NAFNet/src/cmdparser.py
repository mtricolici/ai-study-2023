import argparse
import vars

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="Video enhancer demo")
    parser.add_argument("-i", "--input-file", required=True, help="Path to the input file")
    parser.add_argument("-o", "--output-file", required=True, help="Path to the output file")
    parser.add_argument("-f", "--fps", type=int, default=None, help="Frames per second (default: it will be detected automatically)")
    parser.add_argument("-m", "--model-name", default="NAFNet-REDS-width64", help="Model name (default: NAFNet-REDS-width64)")
    parser.add_argument("-k", "--keep-faces", type=int, default=1, help="Do not enhance faces (default: 1)")
    parser.add_argument("-s", "--sleep", type=float, default=0.0, help="Nr of seconds to sleep between frames (default: 0.0)")
    args = parser.parse_args()
    vars.source_file = args.input_file
    vars.target_file = args.output_file
    vars.fps = args.fps
    vars.model_name = args.model_name
    vars.keep_faces = args.keep_faces != 0
    vars.sleep = args.sleep
