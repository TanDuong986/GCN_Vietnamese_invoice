from gcn.graphVN import Grapher
import cv2
import os
import pandas as pd
import ast

data_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
instance = Grapher('mcocr_public_145013aagqw',data_fd)
G,_,_ = instance.graph_formation()
df = instance.relative_distance()
print(df.head(20))

