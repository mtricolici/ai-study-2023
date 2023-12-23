import argparse
import vars

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="Human faces dataset creator")

    parser.add_argument("-i", "--source-path", required=True, help="Path to folder with raw images")
    parser.add_argument("-o", "--output-path", required=True, help="Path to folder to save faces")
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for processing (default: cpu)")

    args = parser.parse_args()
    vars.source_path = args.source_path
    vars.target_path = args.output_path
    vars.device = args.device
