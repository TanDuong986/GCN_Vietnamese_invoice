from graphVN import Grapher
from model import InvoiceGCN
import torch
import torch.nn.functional as F
import os
import numpy as np
from torch_geometric.utils.convert import from_networkx
import torch_geometric

import numpy as np
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import shutil
import os
import time
from tqdm import tqdm_notebook
from sentence_transformers import SentenceTransformer

save_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/material' #express data file
data_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
img_fd = 'Vietnam_invoice_data/preprocessed_data/images'

sent_model = SentenceTransformer('keepitreal/vietnamese-sbert')
def make_sent_bert_features(text):
    emb = sent_model.encode([text])[0]
    return emb

def cook_input(name):
    '''Get input is name of image, Output is instance of geometry of data'''
    img = cv2.imread(os.path.join(data_fd,'images',name+'.jpg'))
    graph = Grapher(name,data_fd)
    G,_,_ = graph.graph_formation()
    df = graph.relative_distance()
    data = from_networkx(G)
    feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l','line_number', \
            'n_upper', 'n_alpha', 'n_spaces', 'n_numeric','n_special']
    sentence_embbed = np.array(df["content"].map(make_sent_bert_features).tolist()).astype(np.float32)
    numeric_features = df[feature_cols].values.astype(np.float32)
    features = np.concatenate((numeric_features,sentence_embbed),axis = 1)
    features = torch.tensor(features)
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError as e:
            pass
    df['label_text'] = df['label_text'].fillna('OTHER')
    df.loc[df['label_text'] == 'SELLER', 'num_labels'] = 1
    df.loc[df['label_text'] == 'ADDRESS', 'num_labels'] = 2
    df.loc[df['label_text'] == 'TIMESTAMP', 'num_labels'] = 3
    df.loc[df['label_text'] == 'TOTAL_COST', 'num_labels'] = 4
    df.loc[df['label_text'] == 'OTHER', 'num_labels'] = 5

    assert df['num_labels'].isnull().values.any() == False, f'labeling error! Invalid label(s) present in {name}.csv'
    labels = torch.tensor(df['num_labels'].values.astype(np.int8))
    text = df['content'].values

    data.x = features
    data.y = labels
    data.text = text
    data.img_id = name
    
    # instance_geo = torch_geometric.data.Batch.from_data_list(list(data)) # this create batch from instance data 
    return G,img,df,data

#/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/images/mcocr_public_145014zvrla.jpg
if __name__ == "__main__":
    # test_data = torch.load(os.path.join(save_fd, 'test_dataVN.dataset'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G,img,df,test_data = cook_input('mcocr_public_145014zvrla')
    here = os.path.dirname(os.path.abspath(__file__))

    model = InvoiceGCN(input_dim=test_data.x.shape[1], chebnet=True)
    model.load_state_dict(torch.load(os.path.join(here,'weights','model_std.pt')))
    model.eval()

    test_output_fd = os.path.join(here,'output_result')
    # shutil.rmtree(test_output_fd)
    if not os.path.exists(test_output_fd):
        os.mkdir(test_output_fd)
    model.to(device)
    test_data.to(device)

    y_preds = model(test_data).max(dim=1)[1].cpu().numpy() # 0,1,0,2,3,4,... (offset 1)
    LABELS = ["Nguoi ban", "Dia chi", "Ngay", "Thanh toan", "Khac"]
    # test_batch = test_data.batch.cpu().numpy()
    # indexes = range(len(test_data.img_id)) # take all the names of image in dataset, but if only image, it name of image
    index = test_data['img_id']
    start = time.time()
    y_pred = y_preds

    print("Img index: {}".format(index))
    print("y_pred: {}".format(Counter(y_pred)))

    assert len(y_pred) == df.shape[0]

    img2 = np.copy(img)
    for row_index, row in df.iterrows():
        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
        true_label = row["label_text"]

        # if isinstance(true_label, str) and true_label != "invoice": #draw for OTHER
        #     cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _y_pred = y_pred[row_index]
        if _y_pred != 4:
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
            _label = LABELS[_y_pred]
            cv2.putText(
                img2, "{}".format(_label), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

    end = time.time()
    print("\tImage {}: {}".format(index, end - start))
    plt.imshow(img2)
    plt.axis('off')
    plt.savefig(os.path.join(test_output_fd, '{}_result_.png'.format(index)), bbox_inches='tight',dpi = 300)

