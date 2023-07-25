from gcn.graphVN import Grapher
from gcn.model import InvoiceGCN
import torch
import cv2
import os
import pandas as pd
import ast
import tqdm
import time
from tqdm.notebook import tqdm

from tqdm import tqdm
import numpy as np

train_x = np.random.rand(100)


# Notice `train_iter` can only be iter over once, so i get `total` in this way.

with tqdm(total=len(train_x)) as pbar:
    for item in train_x:
        time.sleep(0.2)
        pbar.update(1)
