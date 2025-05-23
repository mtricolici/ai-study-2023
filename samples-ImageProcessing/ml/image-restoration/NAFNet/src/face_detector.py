import os
import warnings
import threading
import cv2
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import insightface
import onnxruntime as ort
import vars

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

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
def get_face_analyser():
  global FACE_ANALYSER
  with THREAD_LOCK:
    if FACE_ANALYSER is None:
      warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform")
      FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=get_execution_provider())
      FACE_ANALYSER.prepare(ctx_id=0)
  return FACE_ANALYSER

#####################################################################
def release_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER:
        del FACE_ANALYSER
    FACE_ANALYSER = None

#####################################################################
def detect_faces(files, threads_count=12):
  print(f"detecting faces in {len(files)} frames ...")
  vars.faces = [None] * len(files)  # Pre-allocate to preserve order

  def process_frame(idx, path):
    img = cv2.imread(path)
    faces = get_face_analyser().get(img)
    coords = [face.bbox.astype(int) for face in faces]
    return idx, coords

  with tqdm(total=len(files)) as pbar:
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
      futures = {executor.submit(process_frame, idx, path): idx for idx, path in enumerate(files)}
      for future in as_completed(futures):
        try:
          idx, coords = future.result()
          vars.faces[idx] = coords
          pbar.update(1)
        except Exception as e:
          print(f"[ERROR] {e}")
          traceback.print_exc()
          executor.shutdown(wait=True, cancel_futures=True)
          os._exit(1)

  release_face_analyser() # Release GPU used VRAM for other models
  gc.collect()
#####################################################################
def demo_face_detector(input_file, output_file):
  print(f'detecting faces in {input_file} ...')


  image = cv2.imread(input_file)
  faces = get_face_analyser().get(image)

  for idx, face in enumerate(faces):
    bbox = face.bbox.astype(int)
    print(f"Face {idx + 1}: {bbox}")
    p1 = (bbox[0], bbox[1])
    p2 = (bbox[2], bbox[3])
    color = (0, 255, 0)
    cv2.rectangle(image, p1, p2, color, 1)

  cv2.imwrite(output_file, image)
  print(f'image with faces marked is saved in {output_file} ;)')
  release_face_analyser()
#####################################################################

if __name__ == "__main__":
  demo_face_detector('/images/src.png', '/images/faces.png')

