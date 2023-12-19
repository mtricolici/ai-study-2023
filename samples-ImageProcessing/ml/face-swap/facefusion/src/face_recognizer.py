import os
import threading
import cv2
import numpy
import onnxruntime

import helper
import vars

FACE_RECOGNIZER = None
THREAD_LOCK = threading.Lock()

#####################################################################
def get_face_recognizer_path():
    model_path = f"/models/face-recognizer/{vars.face_recognizer_model}.onnx"
    if not os.path.exists(model_path):
        print(f"Model {model_path} does not exist :(")
        sys.exit(1)
    return model_path
#####################################################################
def get_face_recognizer():
    global FACE_RECOGNIZER

    with THREAD_LOCK:
        if FACE_RECOGNIZER is None:
            FACE_RECOGNIZER = onnxruntime.InferenceSession(
                get_face_recognizer_path(),
                providers = helper.get_execution_provider())

    return FACE_RECOGNIZER
#####################################################################
def release_face_recognizer():
    global FACE_RECOGNIZER
    FACE_RECOGNIZER = None
#####################################################################

