import os
import sys
import argparse
import vars

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="FaceFussion simplified")
    parser.add_argument('command', choices=['detect', 'swap', 'info'], help='The command to execute')

    parser.add_argument("-i", "--input-file", help="Path to the input file")
    parser.add_argument("-o", "--output-file", help="Path to the output file")
    parser.add_argument("-f", "--face", help="Path to face source file")

    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use for processing (default: cuda)")
    parser.add_argument("--swap-model", default='inswapper_128', help="Face swap Model name (default: inswapper_128)")
    parser.add_argument("--detect-model", default='yunet_2023mar', help="Face detect model name (default: yunet_2023mar)")
    parser.add_argument("--rec-model", default='arcface_w600k_r50', help="Face recognizer model name (default: arcface_w600k_r50)")

    args = parser.parse_args()

    vars.command = args.command

    if vars.command in ("swap", "detect"):

        if args.input_file is None:
            print('Error: input-file is required for this command')
            parser.print_help()
            sys.exit(1)

        if args.output_file is None:
            print('Error: output-file is required for this command')
            parser.print_help()
            sys.exit(1)

        vars.input_file = args.input_file
        vars.output_file = args.output_file

        if vars.command == 'swap':
            if args.face is None:
                print('Error: face is required for this command')
                parser.print_help()
                sys.exit(1)
            vars.face_file = args.face

    vars.device = args.device
    vars.face_swap_model = args.swap_model
    vars.face_detect_model = args.detect_model
    vars.face_recognizer_model = args.rec_model


