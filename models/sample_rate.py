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

    adj = graph.adj(scipy_fmt="csr")

    S = np.array([16,32,64,128,256,512,1024])

    rates = []
    for s in S:
        rates.append(cache_sample_rate(adj, s))

    fname = args.dataset + "_sample_rate.csv"

    with open(fname, 'w') as f:
        f.write(f"{S[0]}")
        for s in S[1:]:
            f.write(f", {s}")
        f.write("\n")
    
        f.write("{:.3f}".format(rates[0]*100))
        for r in rates[1:]:
            f.write(", {:.3f}".format(r*100))