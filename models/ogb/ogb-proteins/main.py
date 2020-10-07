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
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator, NodePropPredDataset

import scipy

import dgl.nn.pytorch as dglnn
import dgl
import json

from torch_sparse import SparseTensor

from tqdm import tqdm, trange

STATE_DICT_PATH = "learned_param_proteins.pt"

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, 
                 norm):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(dglnn.GraphConv(in_feats, n_hidden, norm))
        for _ in range(n_layers):
            self.convs.append(dglnn.GraphConv(n_hidden, n_hidden, norm))
        self.convs.append(dglnn.GraphConv(n_hidden, n_classes, norm))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, feat):
        h = feat
        for conv in self.convs[:-1]:
            h = conv(graph, h)
            h = self.activation(h)
            h = self.dropout(h)
        return self.convs[-1](graph, h)

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, 
                 agg_type):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(dglnn.SAGEConv(in_feats, n_hidden, agg_type))
        for _ in range(n_layers-2):
            self.convs.append(dglnn.SAGEConv(n_hidden, n_hidden, agg_type))
        self.convs.append(dglnn.SAGEConv(n_hidden, n_classes, agg_type))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, feat):
        h = feat
        for conv in self.convs[:-1]:
            h = conv(graph, h)
            h = self.activation(h)
            h = self.dropout(h)
        return self.convs[-1](graph, h)

def warmup_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        lr *= epoch / 50

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def train(model, graph, feat, labels, train_idx, optimizer):
    model.train()
    criterion = th.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = criterion(pred[train_idx], labels[train_idx].to(th.float))
    loss.backward()
    optimizer.step()

    return loss, pred

@th.no_grad()
def evaluate(model, graph, feat, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    y_pred = model(graph, feat)

    train_rocauc = evaluator.eval({
        'y_true': labels[train_idx],
        'y_pred': y_pred[train_idx]
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': labels[val_idx],
        'y_pred': y_pred[val_idx]
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': labels[test_idx],
        'y_pred': y_pred[test_idx]
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

def inference(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat, evaluator):
    if args.sage:
        pass
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.agg_type)
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.norm)
    model = model.to(device)
    model.load_state_dict(th.load(args.state_dict))
    model.eval()
    times = []
    kernel_time = []
        # prof = th.autograd.profiler.profile(use_cuda=True).__enter__()
    with th.autograd.profiler.profile(args.perf, use_cuda=True) as prof:
        for run in tqdm(range(1, args.n_runs + 1)):
            tic = time.time()
            with th.no_grad():
                logits = model(graph, feat)
            toc = time.time()
            times.append(toc - tic)
    if args.perf:
        for evt in prof.key_averages():
            if evt.key == "GSpMM":
                kernel_time.append(evt.cuda_time*evt.count/args.n_runs)
    train_rocauc, valid_rocauc, test_rocauc = evaluate(model, graph, feat, labels, train_idx, val_idx, test_idx, evaluator)
    return valid_rocauc, test_rocauc, times, {"kernel": kernel_time} if prof else None

def run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat, evaluator):
    # Define model and optimizer
    norm = args.norm
    if args.sage:
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.agg_type)
        pass
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, norm)
    model = model.to(device)
    model.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=args.verbose)
    train_rocauc, valid_rocauc, test_rocauc = 0, 0, 0

    with trange(1, args.n_epochs + 1) as t:
        for epoch in t:
            warmup_learning_rate(optimizer, args.lr, epoch)

            loss, pred = train(model, graph, feat, labels, train_idx, optimizer)

            with th.no_grad():
                val_loss = th.nn.BCEWithLogitsLoss()(pred[val_idx], labels[val_idx].to(th.float))

            lr_scheduler.step(val_loss)
            t.set_postfix(train=train_rocauc, val=valid_rocauc, test=test_rocauc)

            if epoch % args.log_every == 0:
                train_rocauc, valid_rocauc, test_rocauc = evaluate(model, graph, feat, labels, train_idx, val_idx, test_idx, evaluator)
                print(f"Train:{train_rocauc:.4f}, Val:{valid_rocauc:.4f}, Test:{test_rocauc:.4f}")
        
    return model



def main():
    argparser = argparse.ArgumentParser("OGBN-Proteins")
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--sage", action="store_true")
    argparser.add_argument("--perf", action="store_true")
    argparser.add_argument("--perf-every", type=int, default=50)
    argparser.add_argument("--perf-file", type=str, default="")
    argparser.add_argument("--n-runs", type=int, default=1)
    argparser.add_argument("--n-epochs", type=int, default=500)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=50)
    argparser.add_argument("--agg-type", type=str, default="mean", help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument("--log-file", type=str, default="", help="Log file name")
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--inference", action="store_true")
    argparser.add_argument("--norm", type=str, default="right")
    argparser.add_argument("--state-dict", type=str, default=STATE_DICT_PATH)
    args = argparser.parse_args()
    print(args)

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)
        th.cuda.set_device(device)
    
    data = DglNodePropPredDataset(name="ogbn-proteins")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    i1, i2, v = graph.edges('all')
    feat = SparseTensor.from_edge_index(th.stack([i2,i1]), graph.edata['feat'][v])
    graph = dgl.graph((i1, i2), idtype=th.int32)
    feat = feat.mean(dim=1)

    # data = NodePropPredDataset(name="ogbn-proteins")
    # graph_raw, labels = data[0]
    # u, v = graph_raw['edge_index']
    # adj = th.sparse.FloatTensor(th.from_numpy(graph_raw['edge_index']), th.from_numpy(graph_raw['edge_feat']))
    # adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
    # feat = adj.mean(dim=1)
    # adj.set_value_(None)
    # adj_t = adj.set_diag()
    # deg = adj_t.sum(dim=1).to(th.float)
    # deg_inv_sqrt = deg.pow(-0.5)
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    # u, v, val = adj_t.coo()
    # adj_t = scipy.sparse.coo_matrix((val.numpy(),(u.numpy(),v.numpy())))
    # graph = dgl.DGLGraph(adj_t)
    

    evaluator = Evaluator(name='ogbn-proteins')
    in_feats = feat.shape[1]

    
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    graph = dgl.to_simple(graph, return_counts=None)

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    n_classes = 112

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    feat = feat.to(device)

    if not args.inference:
        print(f"===== Training... =====")
        model = run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat, evaluator)
        train_rocauc, valid_rocauc, test_rocauc = evaluate(model, graph, feat, labels, train_idx, val_idx, test_idx, evaluator)
        print(f"Train Finished. Best val/test rocauc:{valid_rocauc:.4f}/{test_rocauc:.4f}")
        if args.train:
            th.save(model.state_dict(), args.state_dict)
    else:
        val_accs = []
        test_accs = []
        total_time = []
        stats = {
            "args": vars(args)
        }
        val_acc, test_acc, times, prof = inference(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat, evaluator)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        total_time.extend(times)
        print(f"Inference Finished. Best val/test rocauc:{val_acc:.4f}/{test_acc:.4f}")
        stats[f"gnn"] = {
            "time": total_time,
            "val_accs": val_accs,
            "test_accs": test_accs,
            "kernel_time": prof["kernel"]
        }
        with open(f"{args.log_file}.json", "w") as file:
            file.write(json.dumps(stats, sort_keys=True, indent=4))

if __name__ == "__main__":
    main()
