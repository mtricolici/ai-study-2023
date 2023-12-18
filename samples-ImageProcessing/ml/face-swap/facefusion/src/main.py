import onnxruntime
import torch
import cv2

from cmdparser import parse_cmd_args
from face_detector import detect_all_faces

import vars

#####################################################################
def show_info():
    print(f'Torch gpu available: {torch.cuda.is_available()}')
    providers = onnxruntime.get_available_providers()
    print(f'OnnxRuntime providers: {", ".join(providers)}')

#####################################################################
def demo_face_detect():
    img = cv2.imread(vars.input_file)
    faces = detect_all_faces(img)
    print(f'Found faces: {faces}')

#####################################################################
def main():
    parse_cmd_args()

    if vars.command == 'info':
        show_info()
    elif vars.command == 'detect':
        demo_face_detect()

if __name__ == "__main__":
    main()
