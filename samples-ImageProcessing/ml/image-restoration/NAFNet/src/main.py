import argparse

from cmdparser import parse_cmd_args
import vars

def main():
    parse_cmd_args()

    print(f"Input File: {vars.source_file}")
    print(f"Output File: {vars.target_file}")
    print(f"FPS: {vars.fps}")
    print(f"Model Name: {vars.model_name}")

if __name__ == "__main__":
    main()
