import numpy as np
import cv2
import piq
import torch

#########################################################################
def image_score(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return brisque(img)
#########################################################################
def brisque(img):
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return piq.brisque(img, data_range=255.)
#########################################################################
