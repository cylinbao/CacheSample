import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class ResGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 dropout=0.):
        super(ResGCN, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        # num_layuers = 1 means 2 layers: input and output layer
        for _ in range(num_layers-1):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        self.out_layer = GraphConv(hid_dim, out_dim)

        self.reset_params()

    def reset_params(self):
        self.out_layer.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()

    def forward(self, g, feats, norm_type='right', norm_bias=0, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        h = feats

        for i, layer in enumerate(self.layers):
            h = layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, 
                      S=S, seed=seed, sample_rate=sample_rate)
            h = self.dropout(h)
            if i == 0:
                h_1 = h

        # add the first output f and the final output f
        h = h + h_1
        h = self.out_layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, 
                           S=S, seed=seed, sample_rate=sample_rate)
        return h
