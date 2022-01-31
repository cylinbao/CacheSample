import argparse, time
import numpy as np
import networkx as nx
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from model_utils import save_model, load_model
from cache_sample import cache_sample_rate

def get_max_nnz(csr):
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    nnzs = []
    for i in range(nnode):
        nnz = row_ptr[i+1] - row_ptr[i]
        nnzs.append(nnz)

    return np.max(np.array(nnzs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    args = parser.parse_args()
    print(args)

    if args.dataset == 'cora':
        data = CoraGraphDataset()
        graph = data[0]
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
        graph = data[0]
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
        graph = data[0]
    elif args.dataset == 'reddit':
    	data = RedditDataset(self_loop=True)
    elif args.dataset == 'arxiv':
        data = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/ubuntu/.ogb")
        graph, labels = data[0]
    elif args.dataset == 'proteins':
        data = DglNodePropPredDataset(name="ogbn-proteins", root="/home/ubuntu/.ogb")
        graph, labels = data[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # graph = data[0]
    adj = graph.adj(scipy_fmt="csr")

    max_nnz = get_max_nnz(adj)
    print("Max NNZ = ", max_nnz)
