import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import argparse

here = '/home/dtan/Documents/GCN/GCN_Vietnam'
# /home/dtan/Documents/GCN/GCN_Vietnam/Code/U2Net/output/mcocr_public_145014ubwmd.jpg
def gpr():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str
                        ,default='mcocr_public_145014ubwmd.jpg'
                        ,help="source image")
    return parser.parse_args()

def show(img):
    cv2.imshow("tmp window",img)
    cv2.waitKey(0)

def merge(rgb,mask):
    '''
    rgb : is 3 channels image

    mask : is 2D matrix with values in {0,1}'''
    rgb *= mask[:, :, np.newaxis]
    return rgb