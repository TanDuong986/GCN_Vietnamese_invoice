import timeit
import numpy
import cv2
from PIL import Image
import sys
import os

from utils import read_txt,gen_model,stretch_ROI


current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, "..")
sys.path.append(code_dir)

def cv2image(image): #convert cut image into input of ocr
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    model = gen_model() #1.67 s without cuda | 0.02 with cuda
    img_src = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013alybg.jpg')
    position = read_txt('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013alybg.txt')
    i=0
    for pos in position:
        img = stretch_ROI(pos,img_src)
        s = model.predict(cv2image(img))
        cv2.imwrite(f'/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/ROI/ROI_align/sub_{str(i)}.jpg',img)
        print(f'{str(i)}\t{s}')
        i+=1
    
