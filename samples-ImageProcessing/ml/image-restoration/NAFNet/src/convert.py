import os
import gc
import time
import datetime
import cv2
from collections import OrderedDict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import numpy as np
import onnxruntime as ort
import face_detector
import vars

############################################################################
def load_model():
  return ort.InferenceSession(
    f'/models/{vars.model_name}.onnx',
    providers=["CUDAExecutionProvider"])
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
#    print(f'saving face in memory .. {x1},{y1},{x2},{y2}')
    face = img[y1:y2, x1:x2].copy()
    saved_faces.append(face)

  return saved_faces
############################################################################
def restore_faces(path, index, face_images):
  img = cv2.imread(path)

  for (x1, y1, x2, y2), face_img in zip(vars.faces[index], face_images):
    x1, y1, x2, y2 = clamp_coordinates(x1, y1, x2, y2, img.shape[1], img.shape[0])
#    print(f'restoring face from memory .. {x1},{y1},{x2},{y2}')
    img[y1:y2, x1:x2] = face_img

  cv2.imwrite(path, img)

############################################################################
def load_image_onnx(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # Add batch dim -> [1, 3, H, W]
    return img
############################################################################
def save_image_onnx(img_tensor, path):
    img = img_tensor.squeeze(0)  # [1, 3, H, W] -> [3, H, W]
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    cv2.imwrite(path, img)
############################################################################
def process_single_frame(frame_idx, model, in_path, out_path):

  if vars.keep_faces:
    faces = save_faces_in_memory(in_path, frame_idx)

  img = load_image_onnx(in_path)
  output = model.run(None, {"input": img})[0]
  save_image_onnx(output, out_path)

  del img
  del output
  gc.collect()

  if vars.keep_faces:
    restore_faces(out_path, frame_idx, faces)

############################################################################
def process_frames(threads_count=4):
  files = [os.path.join('/images/tmp',f) for f in os.listdir('/images/tmp/') if f.endswith('.png')]
  files.sort()

  total = len(files)

  if vars.keep_faces:
    face_detector.detect_faces(files)

  model = load_model()

  with tqdm(total=total) as pbar:
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
      futures = {executor.submit(process_single_frame, i, model, files[i], files[i]): i for i in range(total)}
      for future in as_completed(futures):
        frame_index = futures[future]
        try:
          pbar.update(1)
          future.result()
        except Exception as e:
          print(f"[ERROR] Frame {frame_index + 1}: {e}")
          traceback.print_exc()
          executor.shutdown(wait=True, cancel_futures=True)
          os._exit(1)

  print('\nprocess frames finished!')
############################################################################




