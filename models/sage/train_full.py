"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model_utils import save_model, load_model
import dgl.backend.pytorch.sparse as dgl_pytorch_sp


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
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, use_cache_sample=use_cache_sample))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, use_cache_sample=use_cache_sample))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, use_cache_sample=use_cache_sample)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# Run forward and return runtime
def inference(model, graph, features):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(graph, features)
        return time.time() - t0

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type,
                      args.cache_sample)

    if cuda:
        model.cuda()

    if args.inference:
        model_name = "graphsage_{}_agg_{}_n_layer_{}_n_hidden_{}_best.sd".format(
            args.dataset, args.aggregator_type, args.n_layers, args.n_hidden)
        model = load_model(args.dir, model, model_name)
        acc = evaluate(model, g, features, labels, test_nid)
        print("Test accuracy {:.3%}".format(acc))

        num_run = 10
        times = []
        import torch.autograd.profiler as profiler
        with profiler.profile(use_cuda=True) as prof:
            for i in range(num_run):
                t = inference(model, g, features)
                times.append(t)
                print("Inference time: {:.3f}".format(t))
        print("Average inference time: {:.3f}".format(
            np.mean(times[3:])*1000))

        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        events = prof.key_averages()
        for evt in events:
            if evt.key == "GSpMM":
                # print(evt.self_cuda_time_total_str)
                avg_spmm_t = evt.cuda_time*evt.count/num_run/1000
        print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))

        if args.log != "none":
            with open(args.dataset + "_" + args.log + "_log.csv", 'a+') as f:
                if args.cache_sample:
                    S = dgl_pytorch_sp.S
                else:
                    S = 0
                string = "S, {}, ".format(S)
                string += "accuracy, {:.4f}, ".format(acc)
                string += "cuda time, {:.3f}".format(avg_spmm_t)
                f.write(string + "\n")

    if args.train:
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(g, features)
            loss = F.cross_entropy(logits[train_nid], labels[train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, g, features, labels, val_nid)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                acc, n_edges / np.mean(dur) / 1000))

        print()
        acc = evaluate(model, g, features, labels, test_nid)
        print("Test Accuracy {:.4f}".format(acc))

        return acc, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--dir", type=str, default="state_dicts",
            help="directory to store model's state dict")
    parser.add_argument("--train", action='store_true',
            help="perform training")
    parser.add_argument("--inference", action='store_true',
            help="perform inference")
    parser.add_argument("--cache-sample", action='store_true',
            help="Use CacheSample kernel")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    parser.add_argument("--log", type=str, default="none",
            help="filename of log, if none, then no log")
    args = parser.parse_args()
    print(args)

    assert (args.train or args.inference) == True

    if args.train:
        run = 10
        acc_arr = []
        model_arr = []
        for i in range(run):
            acc, model = main(args)
            acc_arr.append(acc)
            model_arr.append(model)
        if args.save_model:
            best_model = model_arr[np.argmax(acc_arr)]
            model_name = "graphsage_{}_agg_{}_n_layer_{}_n_hidden_{}_best.sd".format(
                args.dataset, args.aggregator_type, args.n_layers, args.n_hidden)
            save_model(args.dir, best_model, model_name)
    else:
        main(args)

