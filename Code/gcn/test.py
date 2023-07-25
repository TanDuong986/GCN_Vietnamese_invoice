from graphVN import Grapher
from model import InvoiceGCN
import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch_geometric.utils.convert import from_networkx

import numpy as np
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import shutil
import os
import time


model = InvoiceGCN()

test_output_fd = "/home/dtan/Documents/GCN/test_output"
shutil.rmtree(test_output_fd)
if not os.path.exists(test_output_fd):
    os.mkdir(test_output_fd)

data_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
img_fd = 'Vietnam_invoice_data/preprocessed_data/images'
from tqdm import tqdm_notebook
def make_info(img_id='584'):
    connect = Grapher(img_id, data_fd)
    G, _, _ = connect.graph_formation()
    df = connect.relative_distance()
    individual_data = from_networkx(G)
    img = cv2.imread(os.path.join(img_fd, "{}.jpg".format(img_id)))[:, :, ::-1]

    return G, df, individual_data, img


y_preds = model(test_data).max(dim=1)[1].cpu().numpy()
LABELS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "OTHER"]
test_batch = test_data.batch.cpu().numpy()
indexes = range(len(test_data.img_id))
for index in tqdm_notebook(indexes):
    start = time.time()
    img_id = test_data.img_id[index]  # not ordering by img_id
    sample_indexes = np.where(test_batch == index)[0]
    y_pred = y_preds[sample_indexes]

    print("Img index: {}".format(index))
    print("Img id: {}".format(img_id))
    print("y_pred: {}".format(Counter(y_pred)))
    G, df, individual_data, img = make_info(img_id)

    assert len(y_pred) == df.shape[0]

    img2 = np.copy(img)
    for row_index, row in df.iterrows():
        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
        true_label = row["labels"]

        if isinstance(true_label, str) and true_label != "invoice":
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _y_pred = y_pred[row_index]
        if _y_pred != 4:
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
            _label = LABELS[_y_pred]
            cv2.putText(
                img2, "{}".format(_label), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

    end = time.time()
    print("\tImage {}: {}".format(img_id, end - start))
    plt.imshow(img2)
    plt.savefig(os.path.join(test_output_fd, '{}_result.png'.format(img_id)), bbox_inches='tight')
    plt.savefig('{}_result.png'.format(img_id), bbox_inches='tight')
