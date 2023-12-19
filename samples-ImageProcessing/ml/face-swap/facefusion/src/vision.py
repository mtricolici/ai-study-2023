import cv2
from collections import namedtuple

Face = namedtuple('Face',
[
  'bbox',
  'kps',
  'score',
  'embedding',
  'normed_embedding',
])

#####################################################################
def resize_frame_dimension(frame, max_width, max_height):
  height, width = frame.shape[:2]
  if height > max_height or width > max_width:
    scale = min(max_height / height, max_width / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))
  return frame
#####################################################################

