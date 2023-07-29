from detect_word.inference import inferDetect
from combineLine import end2end,combination
from ocr.utils import gen_model

import time
import numpy as np
import cv2

def draw_poly(canvas,poly):
    color = (0, 0, 255)  # Green color (BGR format)
    thickness = 2  # Thickness of the box's border
    
    for cl in poly:
        xmin, ymin, xmax, ymax = cl
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, thickness)

    cv2.imshow("ve thoi",cv2.resize(canvas,(600,800)))
    cv2.imwrite("done_combination.jpg",canvas)
    cv2.waitKey(0)
if __name__ == "__main__":
    model = gen_model()
    t = time.time()
    path_img = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/images/mcocr_public_145013cxgot.jpg'
    poly = inferDetect(path_img)[0]
    canvas = cv2.imread(path_img)
    poly = np.array(poly).reshape(len(poly),-1)
    box,text = end2end(poly,canvas,model)
    print(box.shape,len(text))
    print(text)
    print(time.time() -t)




