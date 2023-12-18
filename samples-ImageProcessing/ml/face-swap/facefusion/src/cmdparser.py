import os
import argparse
import vars

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="FaceFussion simplified")
    subp = parser.add_subparsers(dest='command')
    subp.required = True

    # define 'info' command without any options
    info = subp.add_parser('info', help='environment info. Is cuda available etc')

    # define 'swap' command with all required options
    swap = subp.add_parser('swap', help='Swap faces on a image or mp4 video')
    swap.add_argument("-i", "--input-file", required=True, help="Path to the input file")
    swap.add_argument("-o", "--output-file", required=True, help="Path to the output file")
    swap.add_argument("-f", "--face", required=True, help="Path to face source file")
    swap.add_argument("-m", "--model", required=True, help="Model name to use. Example: simswap_256, inswapper_128 ...")

    detect = subp.add_parser('detect', help='Detect all faces in a image')
    detect.add_argument("-i", "--input-file", required=True, help="Path to the input file")
    detect.add_argument("-o", "--output-file", required=True, help="Path to the output file")
    detect.add_argument("-m", "--model", required=True, help="Model name to use. Example: retinaface_10g, yunet_2023mar")

    # define arguments for any command
    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use for processing (default: cuda)")
    args = parser.parse_args()

    vars.command = args.command

    if vars.command in ("swap", "detect"):
      vars.input_file = args.input_file
      vars.output_file = args.output_file
      vars.model_name = args.model

      if vars.command == 'swap':
        vars.face_file = args.face
    vars.device = args.device

    print(f'command={vars.command}')
    print(f'if={vars.input_file}')

