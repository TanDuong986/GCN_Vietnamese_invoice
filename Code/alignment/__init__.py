import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import argparse

def show(img):
    cv2.imshow("tmp window",img)
    cv2.waitKey(3000)

def sp(ist):
    print(ist.shape)
