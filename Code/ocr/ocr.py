import timeit
import numpy
import cv2
from PIL import Image

from __init__ import gen_model
from detect_word.perspective import stretch_ROI
from utils import read_txt

def cv2image(image): #convert cut image into input of ocr
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    model = gen_model #1.67 s without cuda | 0.02 with cuda
    img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013amzul.jpg')
    position = read_txt('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013amzul.txt')
    img = stretch_ROI(position[28],img)
    cv2.imshow(img)
    cv2.waitKey(0)