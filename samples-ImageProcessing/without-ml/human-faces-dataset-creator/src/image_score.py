import numpy as np
import cv2
import piq
import torch

#########################################################################
def good_quality_image(img):
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = clip_iqa(img)
    return score >= 0.4
#########################################################################
@torch.no_grad()
def brisque(img):
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)[None, ...] / 255.
    if torch.cuda.is_available():
        img = img.cuda()
    return piq.brisque(img)
#########################################################################
@torch.no_grad()
def clip_iqa(img):
    img = torch.from_numpy(img).permute(2, 0, 1)[None, ...] / 255.
    if torch.cuda.is_available():
        img = img.cuda()
    return piq.CLIPIQA(data_range=1.).to(img.device)(img).item()
#########################################################################
