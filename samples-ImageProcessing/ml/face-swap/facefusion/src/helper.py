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
def distance_to_kps(points, distance):
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]
    kps = numpy.stack((x, y), axis = -1)
    return kps

#####################################################################
def apply_nms(bbox_list, iou_threshold):
  keep_indices = []
  dimension_list = numpy.reshape(bbox_list, (-1, 4))
  x1 = dimension_list[:, 0]
  y1 = dimension_list[:, 1]
  x2 = dimension_list[:, 2]
  y2 = dimension_list[:, 3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  indices = numpy.arange(len(bbox_list))
  while indices.size > 0:
      index = indices[0]
      remain_indices = indices[1:]
      keep_indices.append(index)
      xx1 = numpy.maximum(x1[index], x1[remain_indices])
      yy1 = numpy.maximum(y1[index], y1[remain_indices])
      xx2 = numpy.minimum(x2[index], x2[remain_indices])
      yy2 = numpy.minimum(y2[index], y2[remain_indices])
      width = numpy.maximum(0, xx2 - xx1 + 1)
      height = numpy.maximum(0, yy2 - yy1 + 1)
      iou = width * height / (areas[index] + areas[remain_indices] - width * height)
      indices = indices[numpy.where(iou <= iou_threshold)[0] + 1]
  return keep_indices
#####################################################################

