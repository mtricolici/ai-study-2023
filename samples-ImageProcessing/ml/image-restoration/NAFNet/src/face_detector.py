import warnings
import threading
import cv2
import insightface
import onnxruntime as ort

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

#####################################################################
def get_face_analyser():
  global FACE_ANALYSER
  with THREAD_LOCK:
    if FACE_ANALYSER is None:
      warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform")
      FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
      FACE_ANALYSER.prepare(ctx_id=0)
  return FACE_ANALYSER

#####################################################################
def release_face_analyser():
    global FACE_ANALYSER
    FACE_ANALYSER = None

#####################################################################
def demo_face_detector(input_file, output_file):
  print(f'detecting faces in {input_file} ...')


  image = cv2.imread(input_file)
  faces = get_face_analyser().get(image)

  for idx, face in enumerate(faces):
    bbox = face.bbox.astype(int)
    print(f"Face {idx + 1}: {bbox}")

  release_face_analyser()
#####################################################################

if __name__ == "__main__":
  demo_face_detector('/images/src.png', '/images/faces.png')

