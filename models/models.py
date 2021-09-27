import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import SAGEConv

from cache_sample import sample_rand_coo

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu)) 
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu)) 
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes)) 
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
                 dropout
                 ):
        super(GCNDropEdge, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu)) 
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu)) 
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
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

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, g, features, norm_type='right', norm_bias=0, kernel='cuSPARSE', 
                S=0, seed=None, sample_rate=1.0):
        h = self.dropout(features)
        for l, layer in enumerate(self.layers):
            h = layer(g, h, norm_bias=norm_bias, kernel=kernel, S=S, 
                      seed=seed, sample_rate=sample_rate)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
