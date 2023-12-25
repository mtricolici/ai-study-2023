import warnings
import threading
import cv2
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
    FACE_ANALYSER = None

#####################################################################
def detect_faces(path):
  img = cv2.imread(path)
  if img is None:
      return None, None
  if len(img.shape) != 3 or img.shape[2] != 3:
      # unexpected format. image should be RGB
      return None, None

  faces = get_face_analyser().get(img)
  return img, faces
#####################################################################

