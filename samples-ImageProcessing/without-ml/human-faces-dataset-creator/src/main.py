import os
import time
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
            cv2.imwrite(face_path, face_img)
            face_idx += 1
###########################################################
def main():
    global face_idx
    parse_cmd_args()
    files = list_images()
    total = len(files)

    start_time = time.time()
    last_update_time = start_time

    for i, f in enumerate(files):
        handle_file(f)
        current_time = time.time()

        # print progress every 10 seconds
        if current_time - last_update_time >= 10:
            last_update_time = current_time
            el = time.time() - start_time
            rtime = (total - i - 1) * el / (i + 1)
            p = (i + 1) / total * 100
            print(f"Processed {p:.0f}% {i + 1}/{total} files ({el:.2f} seconds < {rtime:.2f} seconds). Faces found: {face_idx}")
###########################################################

if __name__ == "__main__":
    main()
