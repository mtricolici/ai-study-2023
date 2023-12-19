import cv2
import threading
import numpy
import onnxruntime

import vars
import helper
from face_detector import detect_all_faces, get_one_face

#####################################################################
MODELS =\
{
    'codeformer':
    {
        'path': '/models/face-enh/codeformer.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    },
    'gfpgan_1.2':
    {
        'path': '/models/face-enh/gfpgan_1.2.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    },
    'gfpgan_1.3':
    {
        'path': '/models/face-enh/gfpgan_1.3.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    },
    'gfpgan_1.4':
    {
        'path': '/models/face-enh/gfpgan_1.4.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    },
    'gpen_bfr_256':
    {
        'path': '/models/face-enh/gpen_bfr_256.onnx',
        'template': 'arcface_v2',
        'size': (128, 256)
    },
    'gpen_bfr_512':
    {
        'path': '/models/face-enh/gpen_bfr_512.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    },
    'restoreformer':
    {
        'path': '/models/face-enh/restoreformer.onnx',
        'template': 'ffhq',
        'size': (512, 512)
    }
}
#####################################################################

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()

#####################################################################
def get_face_enhancer():
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            FACE_ENHANCER = onnxruntime.InferenceSession(get_opts('path'), providers = helper.get_execution_provider())
    return FACE_ENHANCER

#####################################################################
def get_opts(name):
    return MODELS.get(vars.face_enh_model).get(name)

#####################################################################
def prepare_crop_frame(frame):
    frame = frame[:, :, ::-1] / 255.0
    frame = (frame - 0.5) / 0.5
    frame = numpy.expand_dims(frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
    return frame
#####################################################################
def normalize_crop_frame(frame):
    frame = numpy.clip(frame, -1, 1)
    frame = (frame + 1) / 2
    frame = frame.transpose(1, 2, 0)
    frame = (frame * 255.0).round()
    frame = frame.astype(numpy.uint8)[:, :, ::-1]
    return frame
#####################################################################
def blend_frame(temp_frame, paste_frame, face_enhancer_blend = 80):
    face_enhancer_blend = 1 - (face_enhancer_blend / 100)
    temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
    return temp_frame

#####################################################################

def enhance_face(face, temp_frame):
    face_enhancer = get_face_enhancer()

    template = get_opts('template')
    msize = get_opts('size')

    crop_frame, affine_matrix = helper.warp_face(temp_frame, face.kps, template, msize)
    crop_frame = prepare_crop_frame(crop_frame)

    face_enhancer_inputs = {}

    for face_enhancer_input in face_enhancer.get_inputs():
        if face_enhancer_input.name == 'input':
            face_enhancer_inputs[face_enhancer_input.name] = crop_frame
        if face_enhancer_input.name == 'weight':
            face_enhancer_inputs[face_enhancer_input.name] = numpy.array([ 1 ], dtype = numpy.double)

    crop_frame = face_enhancer.run(None, face_enhancer_inputs)[0][0]
    crop_frame = normalize_crop_frame(crop_frame)

    face_mask_blur = 0.3 # TODO:??

    paste_frame = helper.paste_back(temp_frame, crop_frame, affine_matrix, face_mask_blur, (0, 0, 0, 0))
    temp_frame = blend_frame(temp_frame, paste_frame)

    return temp_frame
#####################################################################
def process_frame(frame):
    faces = detect_all_faces(frame)
    for face in faces:
        frame = enhance_face(face, frame)
    return frame
#####################################################################
def process_image(input_path, output_path):
    frame = cv2.imread(input_path)
    frame = process_frame(frame)
    cv2.imwrite(output_path, frame)

#####################################################################

