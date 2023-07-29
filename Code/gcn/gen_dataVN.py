import os
from graphVN import Grapher
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.conv import ChebConv,GCNConv
import torch_geometric
from sklearn.utils.class_weight import compute_class_weight

from numpy import random
import tqdm
import numpy as np
import os

here = '/home/dtan/Documents/GCN/GCN_Vietnam'
data_fd ='/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data'
label_pth = os.path.join(data_fd,'label_mcocr2021')
save_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/material'


sent_model = SentenceTransformer('keepitreal/vietnamese-sbert')
def make_sent_bert_features(text):
    emb = sent_model.encode([text])[0]
    return emb

# def get_data(save_fd):
#     """
#     returns one big graph with unconnected graphs with the following:
#     - x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
#     - edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
#     - edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
#     - y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
#     - validation mask, training mask and testing mask
#     """

path = label_pth
files = [i.split('.')[0] for i in os.listdir(path)]
files.sort()
# error_id = unique()
all_files = files[1:] # don't take the first file, this is header
# all_files = list(set(all_files) - set(error_id))
# all_files.sort()

list_of_graphs = []
train_list_of_graphs, test_list_of_graphs = [], []

files = all_files.copy()
random.shuffle(files)

"""Resulting in 550 receipts for training"""
training, testing = files[:872], files[872:] # 80% of 1090

for file in tqdm.tqdm_notebook (all_files):
    connect = Grapher(file, data_fd)
    G,_,_ = connect.graph_formation()
    df = connect.relative_distance()
    individual_data = from_networkx(G)
    # print(df['num_labels'])

    feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l','line_number', \
            'n_upper', 'n_alpha', 'n_spaces', 'n_numeric','n_special']

    text_features = np.array(df["content"].map(make_sent_bert_features).tolist()).astype(np.float32)
    numeric_features = df[feature_cols].values.astype(np.float32)

    features = np.concatenate((numeric_features, text_features), axis=1)
    features = torch.tensor(features)

    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError as e:
            pass

    # df['labels'] = df['labels'].fillna('undefined')
    df.loc[df['label_text'] == 'SELLER', 'num_labels'] = 1
    df.loc[df['label_text'] == 'ADDRESS', 'num_labels'] = 2
    df.loc[df['label_text'] == 'TIMESTAMP', 'num_labels'] = 3
    df.loc[df['label_text'] == 'TOTAL_COST', 'num_labels'] = 4
    df.loc[df['label_text'] == 'OTHER', 'num_labels'] = 5
    # df.loc[df['labels_text'] == 'invoice', 'num_labels'] = 5 # this code do not use 'invoice' label so make it into background

    assert df['num_labels'].isnull().values.any() == False, f'labeling error! Invalid label(s) present in {file}.csv'
    labels = torch.tensor(df['num_labels'].values.astype(np.int8))
    text = df['content'].values

    individual_data.x = features
    individual_data.y = labels
    individual_data.text = text
    individual_data.img_id = file

    if file in training:
        train_list_of_graphs.append(individual_data)
    elif file in testing:
        test_list_of_graphs.append(individual_data)
      # df now is a data frame have 22 columns and 71 row, object is text had embedding to feature
train_data = torch_geometric.data.Batch.from_data_list(train_list_of_graphs)
train_data.edge_attr = None
test_data = torch_geometric.data.Batch.from_data_list(test_list_of_graphs)
test_data.edge_attr = None

torch.save(train_data, os.path.join(save_fd, 'train_dataVN.dataset'))
torch.save(test_data, os.path.join(save_fd, 'test_dataVN.dataset'))
