import numpy
import vars

#####################################################################
def get_execution_provider():
    if vars.device == 'cpu':
        return ['CPUExecutionProvider']
    elif vars.device == 'cuda':
        return ['CUDAExecutionProvider']
    else:
        print('onnxruntime supports only CUDA and CPU :( sorry man.')
        print('Fallback face detector to run on CPU ...')
        return ['CPUExecutionProvider']
#####################################################################
def create_static_anchors(feature_stride, anchor_total, stride_height, stride_width):
    y, x = numpy.mgrid[:stride_height, :stride_width][::-1]
    anchors = numpy.stack((y, x), axis = -1)
    anchors = (anchors * feature_stride).reshape((-1, 2))
    anchors = numpy.stack([ anchors ] * anchor_total, axis = 1).reshape((-1, 2))
    return anchors
#####################################################################
def distance_to_bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    bbox = numpy.column_stack([ x1, y1, x2, y2 ])
    return bbox

#####################################################################

