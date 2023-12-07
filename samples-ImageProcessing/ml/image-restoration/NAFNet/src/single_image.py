from cmdparser import parse_cmd_args
import vars
import convert

def main():
    print('processing single image demo;)')
    parse_cmd_args()
    model = convert.load_model()
    convert.process_single_frame(model, vars.source_file, vars.target_file)
    model = None

if __name__ == "__main__":
    main()
