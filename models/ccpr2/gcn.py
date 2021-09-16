import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from cache_sample import sample_rand_coo

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 # norm
                 ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, # norm=norm, 
            activation=activation)) 
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, # norm=norm, 
                activation=activation)) 
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes)) #, norm=norm)) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, norm_type='right', norm_bias=0, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, S=S, seed=seed, sample_rate=sample_rate)
        return h

class GCNDropEdge(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout
                 ):
        super(GCNDropEdge, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, # norm=norm, 
            activation=activation)) 
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, # norm=norm, 
                activation=activation)) 
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes)) #, norm=norm)) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, norm_type='right', norm_bias=0, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        if sample_rate < 1.0:
            device = features.get_device()
            adj = g.adj(scipy_fmt="coo")
            adj = sample_rand_coo(adj, sample_rate, verbose=False)
            g = dgl.from_scipy(adj, idtype=torch.int32, device=device)
            g = dgl.add_self_loop(g)

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, norm=norm_type, norm_bias=norm_bias, kernel=kernel, S=S, seed=seed)
        return h
