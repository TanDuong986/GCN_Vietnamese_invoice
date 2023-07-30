from detect_word.inference import inferDetect
from combineLine import end2end,combination
from ocr.utils import gen_model
from gcn.use.prepare import cook_input
from gcn.model import InvoiceGCN

import time
import numpy as np
import cv2
import pandas as pd
from collections import Counter
import torch
import os
import matplotlib.pyplot as plt


def draw_poly(canvas,poly):
    color = (0, 0, 255)  # Green color (BGR format)
    thickness = 2  # Thickness of the box's border
    
    for cl in poly:
        xmin, ymin, xmax, ymax = cl
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, thickness)

    cv2.imshow("ve thoi",cv2.resize(canvas,(600,800)))
    cv2.imwrite("done_combination.jpg",canvas)
    cv2.waitKey(0)

def infer_gcn(img,name,data,model):

    out_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/gcn/output_result'
    start = time.time()
    y_preds = model(data).max(dim=1)[1].cpu().numpy() # 0,1,0,2,3,4,... (offset 1)
    print(f'Done infer GCN in {time.time()-start} s')
    LABELS = ["Nguoi ban", "Dia chi", "Ngay", "Thanh toan", "Khac"]
    # test_batch = test_data.batch.cpu().numpy()
    # indexes = range(len(test_data.img_id)) # take all the names of image in dataset, but if only image, it name of image

    # print("y_pred: {}".format(Counter(y_pred)))

    img2 = np.copy(img)
    for row_index, row in df.iterrows():
        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
        _y_pred = y_preds[row_index]
        if _y_pred != 4:
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
            _label = LABELS[_y_pred]
            cv2.putText(
                img2, "{}".format(_label), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

    plt.imshow(img2[:,:,::-1])
    plt.axis('off')
    plt.savefig(os.path.join(out_fd, '{}_result_.png'.format(name)), bbox_inches='tight',dpi = 300)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model = gen_model()
    
    t = time.time()
    path_img = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/data_set_test_craft/46056287dfaa0cf455bb.jpg'
    name_img = os.path.basename(path_img).split(".")[0]

    poly = inferDetect(path_img,cuda=0)[0]
    canvas = cv2.imread(path_img)
    poly = np.array(poly).reshape(len(poly),-1)
    box,text = end2end(poly,canvas,model)
    df_box = pd.DataFrame({"poly":list(box),"content":text},index=None)
    G,df,data = cook_input(canvas,df_box)
    model = InvoiceGCN(input_dim=data.x.shape[1], chebnet=True)
    model.load_state_dict(torch.load(os.path.join(here,'gcn','weights','model_std.pt')))
    model.eval()
    infer_gcn(canvas,name_img,data,model)
    print(time.time()-t)

    







