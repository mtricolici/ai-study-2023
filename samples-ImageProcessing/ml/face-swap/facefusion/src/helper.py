import numpy

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

