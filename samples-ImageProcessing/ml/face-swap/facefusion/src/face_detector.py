import os
import sys
import threading
import cv2
import numpy
import onnxruntime

import helper
import vars
import vision

FACE_DETECTOR = None
THREAD_LOCK = threading.Lock()

#####################################################################
def get_face_detector_model_path():
    model_path = f"/models/face-detect/{vars.face_detect_model}.onnx"
    if not os.path.exists(model_path):
        print(f"Model {model_path} does not exist :(")
        sys.exit(1)
    return model_path

#####################################################################
def get_face_detector():
    global FACE_DETECTOR

    with THREAD_LOCK:
        if FACE_DETECTOR is None:
            if vars.face_detect_model == "yunet_2023mar":
                FACE_DETECTOR = cv2.FaceDetectorYN.create(get_face_detector_model_path(), '', (0, 0))

            elif vars.face_detect_model == "retinaface_10g":
                FACE_DETECTOR = onnxruntime.InferenceSession(
                    get_face_detector_model_path(),
                    providers = helper.get_execution_provider())
            else:
                print(f'Unknown face-detector model {vars.face_detect_model}!')
                sys.exit(1)

    return FACE_DETECTOR

#####################################################################
def release_face_detector():
    global FACE_DETECTOR
    FACE_DETECTOR = None
#####################################################################
def detect_all_faces(frame):
    fd_w = 640
    fd_h = 480

    frame_h, frame_w, _ = frame.shape

    temp = vision.resize_frame_dimension(frame, fd_w, fd_h)
    temp_h, temp_w, _ = temp.shape

    ratio_h = frame_h / temp_h
    ratio_w = frame_w / temp_w

    if vars.face_detect_model == "yunet_2023mar":
        return __detect_faces_yunet(temp, temp_h, temp_w, ratio_h, ratio_w)

    elif vars.face_detect_model == "retinaface_10g":
        return __detect_faces_retinaface(temp, temp_h, temp_w, fd_h, fd_w, ratio_h, ratio_w)

    return []
#####################################################################
def __detect_faces_yunet(frame, height, width, ratio_h, ratio_w):
    face_detector = get_face_detector()
    face_detector.setInputSize((width, height))
    face_detector.setScoreThreshold(0.5)
    bbox_list = []

    _, detections = face_detector.detect(frame)
    if detections.any():
        for detection in detections:
            bbox_list.append(numpy.array(
            [
                detection[0] * ratio_w,
                detection[1] * ratio_h,
                (detection[0] + detection[2]) * ratio_w,
                (detection[1] + detection[3]) * ratio_h
            ]))

    return bbox_list
#####################################################################
def __detect_faces_retinaface(frame, height, width, fd_h, fd_w, ratio_h, ratio_w):
    face_detector_score = 0.5

    face_detector = get_face_detector()
    bbox_list = []
    feature_strides = [ 8, 16, 32 ]
    feature_map_channel = 3
    anchor_total = 2

    prepare_frame = numpy.zeros((fd_h, fd_w, 3))
    prepare_frame[:height, :width, :] = frame

    frame = (prepare_frame - 127.5) / 128.0
    frame = numpy.expand_dims(frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)

    detections = face_detector.run(None,
      {
          face_detector.get_inputs()[0].name: frame
      })

    for index, feature_stride in enumerate(feature_strides):
        keep_indices = numpy.where(detections[index] >= face_detector_score)[0]
        if keep_indices.any():
            stride_height = fd_h // feature_stride
            stride_width = fd_w // feature_stride
            anchors = helper.create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
            bbox_raw = (detections[index + feature_map_channel] * feature_stride)
            kps_raw = detections[index + feature_map_channel * 2] * feature_stride

            for bbox in distance_to_bbox(anchors, bbox_raw)[keep_indices]:
                bbox_list.append(numpy.array(
                [
                    bbox[0] * ratio_width,
                    bbox[1] * ratio_height,
                    bbox[2] * ratio_width,
                    bbox[3] * ratio_height
                ]))

    return bbox_list
#####################################################################

