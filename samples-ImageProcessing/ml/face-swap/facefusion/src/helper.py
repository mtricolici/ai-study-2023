import cv2
import numpy
import vars

#####################################################################
TEMPLATES =\
{
  'arcface_v1': numpy.array(
  [
    [ 39.7300, 51.1380 ],
    [ 72.2700, 51.1380 ],
    [ 56.0000, 68.4930 ],
    [ 42.4630, 87.0100 ],
    [ 69.5370, 87.0100 ]
  ]),
  'arcface_v2': numpy.array(
  [
    [ 38.2946, 51.6963 ],
    [ 73.5318, 51.5014 ],
    [ 56.0252, 71.7366 ],
    [ 41.5493, 92.3655 ],
    [ 70.7299, 92.2041 ]
  ]),
  'ffhq': numpy.array(
  [
    [ 192.98138, 239.94708 ],
    [ 318.90277, 240.1936 ],
    [ 256.63416, 314.01935 ],
    [ 201.26117, 371.41043 ],
    [ 313.08905, 371.15118 ]
  ])
}

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
def warp_face(frame, kps, template, size):
    normed_template = TEMPLATES.get(template) * size[1] / size[0]
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.LMEDS)[0]
    crop_frame = cv2.warpAffine(frame, affine_matrix, (size[1], size[1]), borderMode = cv2.BORDER_REPLICATE)
    return crop_frame, affine_matrix
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
def paste_back(temp_frame, crop_frame, affine_matrix, face_mask_blur, face_mask_padding):
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_size = temp_frame.shape[:2][::-1]
    mask_size = tuple(crop_frame.shape[:2])
    mask_frame = create_static_mask_frame(mask_size, face_mask_blur, face_mask_padding)
    inverse_mask_frame = cv2.warpAffine(mask_frame, inverse_matrix, temp_frame_size).clip(0, 1)
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
    paste_frame = temp_frame.copy()
    paste_frame[:, :, 0] = inverse_mask_frame * inverse_crop_frame[:, :, 0] + (1 - inverse_mask_frame) * temp_frame[:, :, 0]
    paste_frame[:, :, 1] = inverse_mask_frame * inverse_crop_frame[:, :, 1] + (1 - inverse_mask_frame) * temp_frame[:, :, 1]
    paste_frame[:, :, 2] = inverse_mask_frame * inverse_crop_frame[:, :, 2] + (1 - inverse_mask_frame) * temp_frame[:, :, 2]
    return paste_frame
#####################################################################
def create_static_mask_frame(mask_size, face_mask_blur, face_mask_padding):
    mask_frame = numpy.ones(mask_size, numpy.float32)
    blur_amount = int(mask_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    mask_frame[:max(blur_area, int(mask_size[1] * face_mask_padding[0] / 100)), :] = 0
    mask_frame[-max(blur_area, int(mask_size[1] * face_mask_padding[2] / 100)):, :] = 0
    mask_frame[:, :max(blur_area, int(mask_size[0] * face_mask_padding[3] / 100))] = 0
    mask_frame[:, -max(blur_area, int(mask_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        mask_frame = cv2.GaussianBlur(mask_frame, (0, 0), blur_amount * 0.25)
    return mask_frame
#####################################################################

