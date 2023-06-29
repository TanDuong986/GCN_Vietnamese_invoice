import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import argparse

here = '/home/dtan/Documents/GCN/GCN_Vietnam'

def show(img):
    cv2.imshow("tmp window",img)
    cv2.waitKey(0)

def sp(ist):
    print(ist.shape)

def merge(rgb,mask):
    '''
    rgb : is 3 channels image

    mask : is 2D matrix with values in {0,1}'''
    rgb *= mask[:, :, np.newaxis]
    return rgb