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

