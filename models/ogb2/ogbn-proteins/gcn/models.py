import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class GCN(nn.Module):
    def __init__(self, 
                 in_feats, 
                 n_hidden, 
                 n_classes, 
                 n_layers, 
                 activation, 
                 dropout, 
                 norm="both"):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            # GCN with bias can achieve a higher accuracy
            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, norm, bias=bias))
            # self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, norm))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat

        for i in range(self.n_layers):
            h = self.convs[i](graph, h)
            if i < self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 use_cache_sample):
        super(GraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(dglnn.SAGEConv(in_hidden, out_hidden, aggregator_type, use_cache_sample=use_cache_sample))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, graph, feat):
        # h = self.dropout(inputs)
        h = feat
        for l, conv in enumerate(self.convs):
            h = conv(graph, h)
            if l < self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
