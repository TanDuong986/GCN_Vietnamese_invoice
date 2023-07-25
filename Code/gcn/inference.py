from graphVN import Grapher
from model import InvoiceGCN
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.utils.convert import from_networkx

import numpy as np
import os
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
import tqdm

def make_info(img_id,data_fd ='/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data',img_fd ='/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/images'):
    connect = Grapher(img_id, data_fd)
    G, _, _ = connect.graph_formation()
    df = connect.relative_distance()
    img = cv2.imread(os.path.join(img_fd, "{}.jpg".format(img_id)))[:, :, ::-1]
    return G,df,img

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/material'
    test_output_fd = "/home/dtan/Documents/GCN/GCN_Vietnam/Code/gcn/output_result"

    test_data = torch.load(os.path.join(save_fd, 'test_dataVN.dataset'))  
    if not os.path.exists(test_output_fd):
        os.mkdir(test_output_fd)    
    model = InvoiceGCN(input_dim=test_data.x.shape[1],chebnet=True) #778 is sum of 768 [embedding] + 10 [spatial and text properties]
    model.load_state_dict(torch.load(os.path.join(here,'weights','model_std.pt')))
    model.eval()
    model = model.to(device)
    test_data = test_data.to(device)
    
    y_preds = model(test_data).max(dim=1)[1].cpu().numpy()
    LABELS = ["Nguoi ban", "Dia chi", "Ngay ban", "Thanh toan", "Khac"]
    test_batch = test_data.batch.cpu().numpy()
    # an array contains information about location of each cell of each sample
    # [ 0 0 0 0 1 1 1 1 1 2 2 2 2 2 .....] 217 sample, loss 1 sample of testing (null)
    indexes = range(len(test_data.img_id))

    # with tqdm(total=total) as pbar:
    for index in tqdm_notebook(indexes):
        start = time.time()
        img_id = test_data.img_id[index]
        sample_indexes = np.where(test_batch == index)[0]
        y_pred = y_preds[sample_indexes]

        print("Img index: {}".format(index))
        print("Img id: {}".format(img_id))
        print("y_pred: {}".format(Counter(y_pred)))
        G,df,img = make_info(img_id)

        assert len(y_pred) == df.shape[0]

        img2 = np.copy(img)
        for row_index, row in df.iterrows():
            x1,y1,x2,y2 = row[['xmin','ymin','xmax','ymax']]
            true_label = row['label_text']

            # if isinstance(true_label,str) and true_label != "invoice":
            #     cv2.rectangle(img2,(x1,y1),(x2,y2),(0,255,0),2)
            
            y_pred_ = y_pred[row_index]
            if y_pred_ != 4:
                cv2.rectangle(img2,(x1,y1),(x2,y2),(255,0,255),3)
                label_ = LABELS[y_pred_]
                cv2.putText(img2,f'{label_}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            end = time.time()
        print("\tImage {}: {}".format(img_id, end - start))
        plt.imshow(img2)
        plt.axis('off')
        plt.savefig(os.path.join(test_output_fd, '{}_result.png'.format(img_id)), bbox_inches='tight',dpi=300)
        plt.plot


        


