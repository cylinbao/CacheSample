import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class JKNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 dropout=0.):
        super(JKNet, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        # num_layuers = 1 means 2 layers: input and output layer
        for _ in range(num_layers-1):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        cat_dim = hid_dim * (num_layers - 1 + 1)
        self.out_layer = GraphConv(cat_dim, out_dim)

        self.reset_params()

    def reset_params(self):
        self.out_layer.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()

    def forward(self, g, feats, norm_type='right', norm_bias=0, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        h = feats

        feat_lst = []
        for layer in self.layers:
            h = layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, S=S, seed=seed, sample_rate=sample_rate)
            h = self.dropout(h)
            feat_lst.append(h)
        
        # densely concatenate features
        h = torch.cat(feat_lst, dim=-1)
        h = self.out_layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, S=S, seed=seed, sample_rate=sample_rate)

        return h
