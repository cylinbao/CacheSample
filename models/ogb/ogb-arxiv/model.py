#!/usr/bin/env python
# -*- coding: utf-8 -*-
 

import argparse
import math
import random
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset

import dgl.nn.pytorch as dglnn
import dgl
import json

from tqdm import tqdm

class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, norm
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, norm))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        for l in range(self.n_layers):
            h = self.convs[l](graph, h)
            if l < self.n_layers - 1:
                h = self.bns[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h

class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, agg_type, norm
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(dglnn.SAGEConv(in_hidden, out_hidden, agg_type))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        for l in range(self.n_layers):
            h = self.convs[l](graph, h)
            if l < self.n_layers - 1:
                h = self.bns[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
