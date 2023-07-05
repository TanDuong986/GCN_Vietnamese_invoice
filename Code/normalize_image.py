import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],dtype=np.float32,)
    return img

def denormalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

img = io.imread('/home/dtan/Documents/GCN/GCN_Vietnam/34.jpg')
norm_img = normalizeMeanVariance(img)
dnm_image = denormalizeMeanVariance(norm_img)
plt.subplot(1,2,1)
plt.imshow(norm_img)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(dnm_image)
plt.axis("off")
plt.show()