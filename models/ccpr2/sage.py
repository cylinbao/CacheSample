import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, g, features, norm_type='right', norm_bias=1, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        h = self.dropout(features)
        for l, layer in enumerate(self.layers):
            h = layer(g, h, norm_bias=norm_bias, kernel=kernel, S=S, 
                      seed=seed, sample_rate=sample_rate)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
