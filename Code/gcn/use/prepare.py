from gcn.use.graph import Grapher
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.utils.convert import from_networkx

import numpy as np

sent_model = SentenceTransformer('keepitreal/vietnamese-sbert')
def make_sent_bert_features(text):
    emb = sent_model.encode([text])[0]
    return emb

def cook_input(img,df):
    '''Get input is name of image, Output is instance of geometry of data'''
    graph = Grapher(img,df) 
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

    text = df['content'].values
    data.x = features
    data.text = text
    
    # instance_geo = torch_geometric.data.Batch.from_data_list(list(data)) # this create batch from instance data 
    return G,df,data