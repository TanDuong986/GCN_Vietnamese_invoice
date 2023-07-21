import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2

def gen_model():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['cnn']['pretrained'] = True
    config['device'] = 'cuda'
    return Predictor(config) 

def read_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    # Convert the data into a matrix
    matrix = np.zeros((len(lines), 8), dtype=int)
    for i, line in enumerate(lines):
        values = line.strip().split(',') # strip is not take /n, split is its name
        matrix[i] = [int(val) for val in values]
    return matrix

def stretch_ROI(roi,src):  # 0,0016s
    '''Input : roi is coordinate size (8,), src : source image
            
    Output : aligmented image'''
    old_coor = roi.reshape(-1,2)
    x_ = old_coor[:,0]
    y_ = old_coor[:,1]

    xmin = min(x_)
    xmax = max(x_)
    ymin = min(y_)
    ymax = max(y_)

    size = src.shape[0:2] # h,w
    new_coor = np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]).reshape(-1,2)
    #convert matrix 
    cvm = cv2.getPerspectiveTransform(old_coor.astype(np.float32),new_coor.astype(np.float32))
    result = cv2.warpPerspective(src, cvm, size[::-1])
    h,w = int(ymax-ymin),int(xmax-xmin)
    csd = cv2.resize(result[ymin:ymax,xmin:xmax],(w*2,h*2),cv2.INTER_CUBIC)
    return csd