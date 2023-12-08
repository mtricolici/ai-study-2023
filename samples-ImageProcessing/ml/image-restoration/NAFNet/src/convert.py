import os
import time
import datetime
import cv2
import torch
from collections import OrderedDict
import basicsr.models as bm
import basicsr.train as bt
import basicsr.utils as bu
import face_detector
import vars

############################################################################
def load_model():
  model_ops_path = f'/app/{vars.model_name}.yml'
  opt = bt.parse(model_ops_path, is_train=False)
  opt['num_gpu'] = torch.cuda.device_count()
  opt['dist'] = False
  model = bm.create_model(opt)
  return model
############################################################################
def clamp_coordinates(x1, y1, x2, y2, image_width, image_height):
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    return x1, y1, x2, y2

############################################################################
def save_faces_in_memory(path, index):
  img = cv2.imread(path)
  saved_faces = []

  for (x1,y1,x2,y2) in vars.faces[index]:
    x1, y1, x2, y2 = clamp_coordinates(x1, y1, x2, y2, img.shape[1], img.shape[0])
    print(f'saving face in memory .. {x1},{y1},{x2},{y2}')
    face = img[y1:y2, x1:x2].copy()
    saved_faces.append(face)

  return saved_faces
############################################################################
def restore_faces(path, index, face_images):
  img = cv2.imread(path)

  for (x1, y1, x2, y2), face_img in zip(vars.faces[index], face_images):
    x1, y1, x2, y2 = clamp_coordinates(x1, y1, x2, y2, img.shape[1], img.shape[0])
    print(f'restoring face from memory .. {x1},{y1},{x2},{y2}')
    img[y1:y2, x1:x2] = face_img

  cv2.imwrite('/images/restored-faces.png', img)

############################################################################
def process_single_frame(frame_idx, model, in_path, out_path):

  if vars.keep_faces:
    faces = save_faces_in_memory(in_path, frame_idx)

  # read image
  file_client = bu.FileClient('disk')
  img = file_client.get(in_path, None)
  img = bu.imfrombytes(img, float32=True)
  img = bu.img2tensor(img, bgr2rgb=True, float32=True)

  # Feed model
  model.feed_data(data={'lq': img.unsqueeze(dim=0)})
  if model.opt['val'].get('grids', False):
    model.grids()
  model.test()
  if model.opt['val'].get('grids', False):
    model.grids_inverse()

  # Save image
  visuals = model.get_current_visuals()
  img = bu.tensor2img([visuals['result']])
  bu.imwrite(img, out_path)

  if vars.keep_faces:
    restore_faces(out_path, frame_idx, faces)
############################################################################
def process_frames():
  files = [f for f in os.listdir('/images/tmp/') if f.endswith('.png')]
  files.sort()

  total = len(files)

  if vars.keep_faces:
    face_detector.detect_faces(files)

  model = load_model()

  last_print_time = time.time()
  last_print_iterations = 0

  for i, f in enumerate(files, start=1):
    path = os.path.join('/images/tmp', f)
    process_single_frame(i-1, model, path, path)

    time_elapsed = time.time() - last_print_time
    iterations_processed = i - last_print_iterations
    fps = iterations_processed / time_elapsed
    if time_elapsed > 5:
      done = (i / total) * 100.0
      ert = int((total - i) / fps)
      ert = str(datetime.timedelta(seconds=ert))
      print(f'{i} of {total} = {done:.2f} % ({fps:.2f} frames/sec). Will finish in ~ {ert} \r', end='')
      last_print_time = time.time()
      last_print_iterations = i

  print('\nprocess frames finished!')
############################################################################

