from cmdparser import parse_cmd_args
import vars
import convert
import face_detector

def main():
    print('processing single image demo;)')
    parse_cmd_args()

    if vars.keep_faces:
      face_detector.detect_faces([vars.source_file])

    model = convert.load_model()
    convert.process_single_frame(0, model, vars.source_file, vars.target_file)
    model = None

if __name__ == "__main__":
    main()
