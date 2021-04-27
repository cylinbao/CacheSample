import argparse, time
import numpy as np
import scipy as sp
import dgl
import torch as th
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as PyG_T

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../models'))
from model_utils import save_model, load_model
from cache_sample import cache_sample_rand_csr

def get_pubmed():
    data = PubmedGraphDataset()
    return data[0]

def get_arxiv():
    data = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/ubuntu/.ogb")
    graph, labels = data[0]

    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    graph = graph.remove_self_loop().add_self_loop()
    return graph.int()

def get_proteins():
    dataset = PygNodePropPredDataset(name='ogbn-proteins', 
            root="/home/ubuntu/.ogb", transform=PyG_T.ToSparseTensor())
    pyg_data = dataset[0]

    # construct dgl graph from pyg adj mat
    row, col, val = pyg_data.adj_t.coo()
    graph = dgl.graph((row, col), idtype=th.int32)
    labels = pyg_data.y

    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    return graph

def get_reddit():
    data = RedditDataset(self_loop=True)
    return data[0]

def get_cache_sample(g, s_len):
    adj = g.adj(scipy_fmt="csr")
    _adj = cache_sample_rand_csr(adj, s_len)
    _graph = dgl.from_scipy(_adj)
    _graph = _graph.remove_self_loop().add_self_loop()
    return _graph.adj(scipy_fmt="csr")

if __name__ == '__main__':
    # g = get_pubmed()
    # adj = get_cache_sample(g, 16) 
    # sp.io.mmwrite("./datasets/pubmed_s16.mtx", adj, field='integer') 
    #               # , symmetry='symmetric')
    # adj = get_cache_sample(g, 32) 
    # sp.io.mmwrite("./datasets/pubmed_s32.mtx", adj, field='integer') 

    # g = get_arxiv()
    # adj = get_cache_sample(g, 16) 
    # sp.io.mmwrite("./datasets/ogbn-arxiv_s16.mtx", adj, field='integer') 
    # adj = get_cache_sample(g, 32) 
    # sp.io.mmwrite("./datasets/ogbn-arxiv_s32.mtx", adj, field='integer') 

    # g = get_proteins()
    # adj = get_cache_sample(g, 128) 
    # sp.io.mmwrite("./datasets/ogbn-proteins_s128.mtx", adj, field='integer') 
    # adj = get_cache_sample(g, 256) 
    # sp.io.mmwrite("./datasets/ogbn-proteins_s256.mtx", adj, field='integer') 

    g = get_reddit()
    adj = get_cache_sample(g, 64) 
    sp.io.mmwrite("./datasets/reddit_s64.mtx", adj, field='integer') 
    # g = get_reddit()
    # adj = get_cache_sample(g, 128) 
    # sp.io.mmwrite("./datasets/reddit_s128.mtx", adj, field='integer') 
    # adj = get_cache_sample(g, 256) 
    # sp.io.mmwrite("./datasets/reddit_s256.mtx", adj, field='integer') 
