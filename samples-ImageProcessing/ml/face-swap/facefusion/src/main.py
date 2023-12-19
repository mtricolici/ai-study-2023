import torch
import onnxruntime as ort
import cv2

from cmdparser import parse_cmd_args
from face_detector import detect_all_faces
import face_swapper

import vars

#####################################################################
def show_info():
    print(f'Torch gpu available: {torch.cuda.is_available()}')
    providers = ort.get_available_providers()
    print(f'OnnxRuntime providers: {", ".join(providers)}')
    cv2_gpus = cv2.cuda.getCudaEnabledDeviceCount()
    print(f'cv2 visible cuda devices: {cv2_gpus}')

#####################################################################
def demo_face_detect():
    img = cv2.imread(vars.input_file)
    faces = detect_all_faces(img)
    for face in faces:
#        print(f'Foud face: {face}')
        box = face.bbox
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(vars.output_file, img)
    print(f'Image with marked faces saved in {vars.output_file} ;)')
#####################################################################
def demo_face_swap():
    face_swapper.process_image(
        vars.face_file,
        vars.input_file,
        vars.output_file)

    print(f'Image with swapped faces saved in {vars.output_file} ;)')

#####################################################################
def main():
    parse_cmd_args()

    if vars.command == 'info':
        show_info()
    elif vars.command == 'detect':
        demo_face_detect()
    elif vars.command == 'swap':
        demo_face_swap()

if __name__ == "__main__":
    main()
