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
import torch.autograd.profiler as profiler


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 kernel="cuSPARSE",
                 S=128):
                 # use_cache_sample):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, 
            kernel=kernel, S=S))
            # use_cache_sample=use_cache_sample))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, 
                kernel=kernel, S=S))
                #use_cache_sample=use_cache_sample))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, 
            kernel=kernel, S=S))
            # use_cache_sample=use_cache_sample)) # activation None

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

def main(args, n_running, name_base):
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
                      args.kernel,
                      args.S)
                      # args.cache_sample)

    if cuda:
        model.cuda()

    if args.inference:
        model_name = name_base + "_best.pt"
        model = load_model(args.dir, model, model_name)  
        acc = evaluate(model, g, features, labels, test_nid)
        print("Test accuracy {:.3%}".format(acc))  

        warm_up = 0
        num_run = 100
        times = []

        for i in range(warm_up):
            inference(model, g, features)

        # with profiler.profile(use_cuda=True) as prof:
        #     for i in range(num_run):
        #         t = inference(model, g, features)
        #         times.append(t)
        #         # print("Inference time: {:.3f}".format(t))
        # # print("Average inference time: {:.3f}".format(np.mean(times[3:])*1000))
        # avg_t = np.mean(times[3:])*1000
        # print("Average inference time: {:.3f}".format(avg_t))

        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        # events = prof.key_averages()
        # for evt in events:
        #     if evt.key == "GSpMM":
        #         # print(evt.self_cuda_time_total_str)
        #         avg_spmm_t = evt.cuda_time*evt.count/num_run/1000
        # print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))

        spmm_t_arr = []
        for i in range(num_run):
            with profiler.profile(use_cuda=True) as prof:
                t = inference(model, g, features)
                times.append(t)

            events = prof.key_averages()
            for evt in events:
                if evt.key == "GSpMM": 
                    spmm_t = evt.cuda_time*evt.count/1000 
                    spmm_t_arr.append(spmm_t)

        print("Average inference time: {:.3f}".format(np.mean(times[3:])*1000))
        avg_spmm_t = np.mean(spmm_t_arr)
        std_spmm_t = np.std(spmm_t_arr)
        print("GSpMM CUDA kernel, Avg Time: {:.3f} (ms), Std: {:.3f}".format(avg_spmm_t, std_spmm_t))

        if args.log != "none":
            with open("./log/" + args.dataset + "_" + args.log + "_log.csv", 'a+') as f:
                if args.kernel == "CacheSample":
                    S = args.S
                else:
                    S = 0
                string = "S, {}, ".format(S)
                string += "accuracy, {:.4f}, ".format(acc)
                string += "avg cuda time, {:.3f}, ".format(avg_spmm_t)
                string += "avg total time, {:.3f}".format(avg_t)
                f.write(string + "\n")
        return 

    if args.train:
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        best_val_acc = 0
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
            
            if best_val_acc < acc:
                best_val_acc = acc

                if args.save_model:
                    model_name = name_base + "_best_{}.pt".format(n_running)
                    save_model(args.dir, model, model_name)

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
    parser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Define kernel from cuSPARSE and CacheSample")
    parser.add_argument("--S", type=int, default=128,
            help="Define S value for CacheSample kernel")
    # parser.add_argument("--cache-sample", action='store_true',
    #         help="Use CacheSample kernel")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    parser.add_argument("--log", type=str, default="none",
            help="filename of log, if none, then no log")
    parser.add_argument("--n-runs", type=int, default=10,
            help="filename of log, if none, then no log")
    args = parser.parse_args()
    print(args)

    assert (args.train or args.inference) == True

    test_accs = []
    name_base = "sage_{}_{}_agg_{}_layer_{}_hidden".format(args.dataset, 
                args.aggregator_type, args.n_layers, args.n_hidden)
    if args.train:
        for i in range(args.n_runs):
            acc, model = main(args, i, name_base)
            test_accs.append(acc)
        
        if args.save_model:
            best_idx = np.argmax(test_accs)
            print("best_idx: ", best_idx)
            print("best test acc: ", test_accs[best_idx])
            model_name = name_base + "_best"
            cmd = "cp {}/{}_{}.pt {}/{}.pt".format(args.dir, model_name, best_idx, args.dir, model_name)
            os.system(cmd)
            # os.system("rm ./{}/{}_*.pt".format(args.dir, model_name))
    else:
        main(args, 0, name_base)

