from torch_geometric.nn.conv import ChebConv,GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class InvoiceGCN(nn.Module):

    def __init__(self, input_dim, chebnet=False, n_classes=5, dropout_rate=0.2, K=3):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        if chebnet:
            self.conv1 = ChebConv(self.input_dim, 64, K=K)
            self.conv2 = ChebConv(64, 32, K=K)
            self.conv3 = ChebConv(32, 16, K=K)
            self.conv4 = ChebConv(16, self.n_classes, K=K)
        else:
            self.conv1 = GCNConv(self.first_dim, 64, improved=True, cached=True)
            self.conv2 = GCNConv(64, 32, improved=True, cached=True)
            self.conv3 = GCNConv(32, 16, improved=True, cached=True)
            self.conv4 = GCNConv(16, self.n_classes, improved=True, cached=True)

    def forward(self, data):
        # for transductive setting with full-batch update
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.conv3(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
    

################################# different model #######################

class GraphConvolution(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.2,
        bias=True,
        activation=F.relu,
    ):
        super().__init__()

        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.0

        self.bias = bias
        self.activation = activation

        def glorot(shape, name=None):
            """Glorot & Bengio (AISTATS 2010) init."""
            init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
            init = torch.FloatTensor(shape[0], shape[1]).uniform_(
                -init_range, init_range
            )
            return init

        self.weight = nn.Parameter(glorot((input_dim, output_dim)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        # node feature, adj matrix
        # D^(-1/2).A.D^(-1/2).H_i.W_i
        # with H_0 = X (init node features)
        # V, A
        x, support = inputs

        x = F.dropout(x, self.dropout)
        xw = torch.mm(x, self.weight)
        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        if self.activation is None:
            return out, support
        else:
            return self.activation(out), support

class LinearEmbedding(torch.nn.Module):

    def __init__(self, input_size, output_size, use_act="relu"):
        super().__init__()
        self.C = output_size
        self.F = input_size

        self.W = nn.Parameter(torch.FloatTensor(self.F, self.C))
        self.B = nn.Parameter(torch.FloatTensor(self.C))

        if use_act == "relu":
            self.act = torch.nn.ReLU()
        elif use_act == "softmax":
            self.act = torch.nn.Softmax(dim=-1)
        else:
            self.act = None

        nn.init.xavier_normal_(self.W)
        nn.init.normal_(self.B, mean=1e-4, std=1e-5)

    def forward(self, V):
        # V shape B,N,F
        # V: node features
        V_out = torch.matmul(V, self.W) + self.B
        if self.act:
            V_out = self.act(V_out)

        return V_out
        
class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64],
                 bias=True, dropout_rate=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.bias = bias
        self.dropout_rate = dropout_rate

        gcn_layers = []
        for index, (h1, h2) in enumerate(
            zip(self.hidden_dims[:-1], self.hidden_dims[1:])):
            gcn_layers.append(
                GraphConvolution(
                    h1,
                    h2,
                    activation=None if index == len(self.hidden_dims) else F.relu,
                    bias=self.bias,
                    dropout=self.dropout_rate,
                    is_sparse_inputs=False
                )
            )

        self.layers = nn.Sequential(*gcn_layers)
        self.linear1 = LinearEmbedding(input_dim, self.hidden_dims[0], use_act='relu')
        self.linear2 = LinearEmbedding(self.hidden_dims[-1], self.output_dim, use_act='relu')

    def forward(self, inputs):
        # features, adj
        x, support = inputs
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout_rate)
        x, _ = self.layers((x, support))
        x = self.linear2(x)
        return x, support
