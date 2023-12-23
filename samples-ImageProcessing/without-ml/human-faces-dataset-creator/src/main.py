import os
import argparse
import cv2
import numpy as np

from cmdparser import parse_cmd_args
import vars
import face_detector

###########################################################
face_idx = 1
###########################################################
def list_images():
   paths = []
   for root, _, files in os.walk(vars.source_path):
       for f in files:
           if f.endswith(('.jpg', '.png')):
               paths.append(os.path.join(root, f))

   return paths
###########################################################
def downscale_image(img, size=128):
    if img.shape[0] < size or img.shape[1] < size:
        return None # Ignore small bad quality images

    ar = img.shape[1] / img.shape[0] # Aspect Ratio
    if ar > 1:
        w  = size
        h = int(size / ar)
    else:
        w  = int(size * ar)
        h = size

    img = cv2.resize(img, (w, h))
    canvas = np.zeros((size, size, 3), dtype='uint8')

    x = int((size - w) / 2)
    y = int((size - h) / 2)

    canvas[ y:y+h, x:x+w ] = img

    return canvas

###########################################################
def handle_file(path):
    global face_idx

    print(f'handling raw file: {path}')
    img, faces = face_detector.detect_faces(path)
    for face in faces:
        x1, y1, x2, y2 = face
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        face_img = img[y1:y2, x1:x2]

        face_img = downscale_image(face_img)
        if face_img is not None:
            face_path = os.path.join(vars.target_path, f'{face_idx:05d}-face.png')
            print(f'--new face {face_path}')
            cv2.imwrite(face_path, face_img)
            face_idx += 1
###########################################################
def main():
    parse_cmd_args()
    files = list_images()
    total = len(files)

    for i, f in enumerate(files):
        handle_file(f)
        if i > 10:
            break
###########################################################

if __name__ == "__main__":
    main()
