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


save_fd = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/material'
def load_train_test_split(save_fd):
    train_data = torch.load(os.path.join(save_fd, 'train_data.dataset'))
    test_data = torch.load(os.path.join(save_fd, 'test_data.dataset'))
    return train_data, test_data

train_data, test_data = load_train_test_split(save_fd=save_fd)
# print(train_data, test_data)

model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.9
)
train_data = train_data.to(device)
test_data = test_data.to(device)

# class weights for imbalanced data
_class_weights = compute_class_weight(
    class_weight="balanced", classes = train_data.y.unique().cpu().numpy(), y= train_data.y.cpu().numpy()
)
print(_class_weights)

no_epochs = 2000
for epoch in range(1, no_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # NOTE: just use boolean indexing to filter out test data, and backward after that!
    # the same holds true with test data :D
    # https://github.com/rusty1s/pytorch_geometric/issues/1928
    loss = F.nll_loss(
        model(train_data), (train_data.y - 1).to(torch.int64), weight=torch.FloatTensor(_class_weights).to(device)
    )
    loss.backward()
    optimizer.step()

    # calculate acc on 5 classes
    with torch.no_grad():
        if epoch % 200 == 0:
            model.eval()

            # forward model
            for index, name in enumerate(['train', 'test']):
                _data = eval("{}_data".format(name))
                y_pred = model(_data).max(dim=1)[1]
                y_true = (_data.y - 1)
                acc = y_pred.eq(y_true).sum().item() / y_pred.shape[0]

                y_pred = y_pred.cpu().numpy()
                y_true = y_true.cpu().numpy()
                print("\t{} acc: {}".format(name, acc))
                # confusion matrix
                if name == 'test':
                    cm = confusion_matrix(y_true, y_pred)
                    class_accs = cm.diagonal() / cm.sum(axis=1)
                    print(classification_report(y_true, y_pred))

            loss_val = F.nll_loss(model(test_data), (test_data.y - 1).to(torch.int64))
            fmt_log = "Epoch: {:03d}, train_loss:{:.4f}, val_loss:{:.4f}"
            print(fmt_log.format(epoch, loss, loss_val))
            print(">" * 50)
        if epoch % 1000:
            torch.save(model.state_dict(),save_fd+f'/KIE_epochs_{str(epoch)}.pt')

