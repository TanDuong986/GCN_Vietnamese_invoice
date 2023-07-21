from gcn.graphVN import Grapher
import cv2
import os
import pandas as pd
import ast

data_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
instance = Grapher('145013geoih',data_fd)
dff = instance.graph_formation()
# path_csv = '/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/convert_data/mcocr_public_145013geoih.jpg.csv'
# with open(path_csv,'r') as f:
#     df = pd.read_csv(f)
print(dff.head(5))
# vd = ast.literal_eval(instance.df.loc[0,"poly"])


