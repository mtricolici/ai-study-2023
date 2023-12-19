import threading
import numpy
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

import vars
import helper
from face_detector import detect_all_faces, get_one_face

#####################################################################
MODELS = \
{
    'blendswap_256':
    {
        'type': 'blendswap',
        'path': '/models/face-swap/blendswap_256.onnx',
        'template': 'ffhq',
        'size': (512, 256),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128':
    {
        'type': 'inswapper',
        'path': '/models/face-swap/inswapper_128.onnx',
        'template': 'arcface_v2',
        'size': (128, 128), 
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128_fp16':
    {
        'type': 'inswapper',
        'path': '/models/face-swap/inswapper_128_fp16.onnx',
        'template': 'arcface_v2',
        'size': (128, 128), 
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'simswap_256':
    {
        'type': 'simswap',
        'path': '/models/face-swap/simswap_256.onnx',
        'template': 'arcface_v1',
        'size': (112, 256),
        'mean': [ 0.485, 0.456, 0.406 ],
        'standard_deviation': [ 0.229, 0.224, 0.225 ]
    },
    'simswap_512_unofficial':
    {
        'type': 'simswap',
        'path': '/models/face-swap/simswap_512_unofficial.onnx',
        'template': 'arcface_v1',
        'size': (112, 512),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    }
}
#####################################################################

FRAME_PROCESSOR = None
MODEL_MATRIX = None
THREAD_LOCK = threading.Lock()

#####################################################################
def get_frame_processor():
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            FRAME_PROCESSOR = onnxruntime.InferenceSession(get_opts('path'), providers = helper.get_execution_provider())

    return FRAME_PROCESSOR
#####################################################################
def get_model_matrix():
    global MODEL_MATRIX

    with THREAD_LOCK:
        if MODEL_MATRIX is None:
            model = onnx.load(get_opts('path'))
            MODEL_MATRIX = numpy_helper.to_array(model.graph.initializer[-1])
    return MODEL_MATRIX
#####################################################################
def get_opts(name):
    return MODELS.get(vars.face_swap_model).get(name)
#####################################################################
def swap_face(source_face, target_face, frame):
    frame_processor = get_frame_processor()

    model_type = get_opts('type')

    crop_frame, affine_matrix = helper.warp_face(frame, target_face.kps, get_opts('template'), get_opts('size'))
    crop_frame = prepare_crop_frame(crop_frame)
    frame_processor_inputs = {}

    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == 'source':
            if model_type == 'blendswap':
                frame_processor_inputs[frame_processor_input.name] = prepare_source_frame(source_face)
            else:
                frame_processor_inputs[frame_processor_input.name] = prepare_source_embedding(source_face)
        if frame_processor_input.name == 'target':
            frame_processor_inputs[frame_processor_input.name] = crop_frame

    crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    crop_frame = normalize_crop_frame(crop_frame)

    face_mask_blur = 0.3 # ???
    face_mask_padding = [ 0, 0, 0, 0 ] # ???

    return helper.paste_back(frame, crop_frame, affine_matrix, face_mask_blur, face_mask_padding)

#####################################################################
def prepare_crop_frame(frame):
    model_mean = get_opts('mean')
    model_standard_deviation = get_opts('standard_deviation')
    frame = frame[:, :, ::-1] / 255.0
    frame = (frame - model_mean) / model_standard_deviation
    frame = frame.transpose(2, 0, 1)
    frame = numpy.expand_dims(frame, axis = 0).astype(numpy.float32)
    return frame
#####################################################################
def prepare_source_frame(face):
    frame = cv2.imread(vars.face_file)
    frame, _ = helper.warp_face(frame, face.kps, 'arcface_v2', (112, 112))
    frame = frame[:, :, ::-1] / 255.0
    frame = frame.transpose(2, 0, 1)
    frame = numpy.expand_dims(frame, axis = 0).astype(numpy.float32)
    return frame

#####################################################################
def prepare_source_embedding(face):
    model_type = get_opts('type')
    if model_type == 'inswapper':
      mmatrix = get_model_matrix()
      embedding = face.embedding.reshape((1, -1))
      embedding = numpy.dot(embedding, mmatrix) / numpy.linalg.norm(embedding)
    else:
      embedding = face.normed_embedding.reshape(1, -1)
    return embedding

#####################################################################
def normalize_crop_frame(frame):
    frame = frame.transpose(1, 2, 0)
    frame = (frame * 255.0).round()
    frame = frame[:, :, ::-1].astype(numpy.uint8)
    return frame

#####################################################################
def process_frame(source_face, frame):
    faces = detect_all_faces(frame)
    for target_face in faces:
      frame = swap_face(source_face, target_face, frame)
    return frame
#####################################################################
def process_image(source_path, target_path, output_path):
  source_face = get_one_face(cv2.imread(source_path))
  frame = cv2.imread(target_path)
  frame = process_frame(source_face, frame)
  cv2.imwrite(output_path, frame)
#####################################################################

