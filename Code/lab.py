from gcn.graphVN import Grapher
from gcn.model import InvoiceGCN
import torch
import cv2
import os
import pandas as pd
import ast

# data_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
# instance = Grapher('mcocr_public_145013aagqw',data_fd)
# G,_,_ = instance.graph_formation()
# df = instance.relative_distance()
# print(df.head(20))
save_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/material'
test_data = torch.load(os.path.join(save_fd, 'test_dataVN.dataset'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = InvoiceGCN(input_dim=test_data.x.shape[1],chebnet=True)
model.load_state_dict(torch.load('/home/dtan/Documents/GCN/GCN_Vietnam/Code/gcn/weights/model_std.pt'))
model.eval()
model.to(device)
test_data.to(device)

# y_pred = model(test_data).max(dim=1)[1].cpu().numpy()
print(test_data[1]['text'])

