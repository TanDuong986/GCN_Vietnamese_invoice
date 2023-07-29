import timeit
import numpy
import cv2
from PIL import Image
import sys
import os

from ocr.utils import read_txt,gen_model,stretch_ROI



def cv2image(image): #convert cut image into input of ocr
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def ocr_csd(pos,img,model):
    # img = stretch_ROI(pos,img_src)
    off = 3
    csd = img[pos[1]-off:pos[3]+off,pos[0]-off:pos[2]+off]
    text = model.predict(cv2image(csd))
    return text

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(current_dir, "..")
    sys.path.append(code_dir)
    model = gen_model()
    img_src = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013alybg.jpg')
    position = read_txt('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013alybg.txt')
    i=0
    for pos in position:
        img = stretch_ROI(pos,img_src)
        s = model.predict(cv2image(img))
        cv2.imwrite(f'/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/ROI/ROI_align/sub_{str(i)}.jpg',img)
        print(f'{str(i)}\t{s}')
        i+=1
    
