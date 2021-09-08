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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import torch.autograd.profiler as profiler
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model_utils import save_model, load_model, EarlyStopping, BestVal

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
            # kernel=kernel, S=S))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
                # kernel=kernel, S=S))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))
            # kernel=kernel, S=S))

    def forward(self, graph, inputs, kernel="cuSPARSE", S=0, seed=None):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h, kernel, S, seed=seed)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid, kernel="cuSPARSE", S=0):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, kernel, S)
        logits = logits[nid]
        labels = labels[nid]
        loss = F.cross_entropy(logits, labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return loss.item(), acc

# Run forward and return runtime
def inference(model, graph, features, kernel="cuSPARSE", S=0):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(graph, features, kernel, S)
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
                      args.aggregator_type)

    if cuda:
        model.cuda()

    if args.inference:
        model_name = name_base + "_best.pt"
        model = load_model(args.dir, model, model_name, args.gpu)
        # loss, acc = evaluate(model, g, features, labels, test_nid, args.kernel, args.S)
        loss, acc = evaluate(model, g, features, labels, test_nid, 'cuSPARSE', 0)
        print("Test accuracy {:.3%}".format(acc))  

        warm_up = 2
        num_run = 10
        times = []

        for i in range(warm_up):
            inference(model, g, features, args.kernel, args.S)

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
                t = inference(model, g, features, args.kernel, args.S)
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

        # if args.log != "none":
        #     with open("./log/" + args.dataset + "_" + args.log + "_log.csv", 'a+') as f:
        #         if args.kernel == "CacheSample":
        #             S = args.S
        #         else:
        #             S = 0
        #         string = "S, {}, ".format(S)
        #         string += "accuracy, {:.4f}, ".format(acc)
        #         string += "avg cuda time, {:.3f}, ".format(avg_spmm_t)
        #         string += "avg total time, {:.3f}".format(avg_t)
        #         f.write(string + "\n")
        return 

    if args.train:
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        model_name = name_base + "_{}.pt".format(n_running)
        if args.early_stop is True:
            early_stop = EarlyStopping(path=args.dir, fname=model_name, verbose=False)
        if args.best_val is True:
            best_val = BestVal(path=args.dir, fname=model_name)

        # initialize graph
        dur = []
        best_val_acc = 0
        for epoch in range(args.n_epochs):
            model.train()
            # if epoch >= 3:
            t0 = time.time()
            seed = int((t0 - math.floor(t0))*1e7)
            # forward
            logits = model(g, features, kernel=args.kernel, S=args.S, seed=seed)
            loss = F.cross_entropy(logits[train_nid], labels[train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # if epoch >= 3:
            dur.append(time.time() - t0)

            val_loss, val_acc = evaluate(model, g, features, labels, val_nid, 
                                    kernel=args.kernel, S=args.S)

            print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | Accuracy {:.5f} | "
                  .format(epoch, dur[-1]*1000, loss.item(), val_acc))

            if args.early_stop is True:
                early_stop(val_loss, model)

                if early_stop.early_stop:
                    print("Early stopping.")
                    model.load_state_dict(early_stop.load_checkpoint(args.gpu))
                    break

            if args.best_val is True:
                best_val(val_loss, model)

        if args.best_val is True:
            model.load_state_dict(best_val.load_checkpoint(args.gpu))

        print()
        val_loss, val_acc = evaluate(model, g, features, labels, val_nid, kernel='cuSPARSE', S=0)
        print("Val Accuracy {:.5f}".format(val_acc))
        test_loss, test_acc = evaluate(model, g, features, labels, test_nid, kernel='cuSPARSE', S=0)
        print("Test Accuracy {:.5f}".format(test_acc))

        epoch_t = np.mean(dur[3:])*1000
        print("Total epoch time (ms): {:.3f}".format(np.sum(dur)))
        print("Mean epoch time (ms): {:.3f}".format(epoch_t))

        if args.save_model:
            model_name = name_base + "_{}.pt".format(n_run)
            save_model(args.dir, model, model_name)

        with torch.no_grad():
            torch.cuda.empty_cache()

        return test_acc, epoch_t


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
    parser.add_argument("--S", type=int, default=0,
            help="Define S value for CacheSample kernel")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    parser.add_argument("--early-stop", action='store_true',
            help="whether to early stoearly stopp")
    parser.add_argument("--best-val", action='store_true',
            help="keep the best validation model")
    parser.add_argument("--log", action='store_true', help="log or not")
    parser.add_argument("--n-runs", type=int, default=10,
            help="filename of log, if none, then no log")
    args = parser.parse_args()

    midle_dir = "{}_S{}".format(args.kernel, args.S)
    # midle_dir = "cuSPARSE_S0"
    args.dir = args.dir + '/' + args.dataset + '/' + midle_dir 

    print(args)

    assert (args.train or args.inference) == True

    name_base = "sage_{}_{}_agg_{}_layer_{}_hidden".format(args.dataset, 
                args.aggregator_type, args.n_layers, args.n_hidden)

    test_accs = []
    epoch_times = []
    if args.train:
        for i in range(args.n_runs):
            acc, epoch_t = main(args, i, name_base)
            test_accs.append(acc)
            epoch_times.append(epoch_t)

        print()
        print(f"Test Accs: {test_accs}")
        print(f"Best Test Accuracy: {np.max(test_accs):.3%}")
        print(f"Average Test accuracy: {np.mean(test_accs):.3%} ± {np.std(test_accs):.3%}")
        print(f"Mean Epoch Time: {np.mean(epoch_times):.3f}")

        if args.save_model or args.early_stop or args.best_val:
            best_idx = np.argmax(test_accs)
            print("best_idx: ", best_idx)
            model_name = name_base 
            cmd = "cp {}/{}_{}.pt {}/{}_best.pt".format(args.dir, model_name, best_idx, args.dir, model_name)
            print(cmd)
            os.system(cmd)
            for i in range(args.n_runs):
                cmd = "rm {}/{}_{}.pt".format(args.dir, model_name, i)
                print(cmd)
                os.system(cmd)

        if args.log:
            log_path = "./log/{}".format(args.dataset)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_name = "{}/sage_{}_mean_agg_{}_S{}_train_log.csv".format(log_path, 
                        args.dataset, args.kernel, args.S)
            with open(log_name, 'a+') as f:
                string = "n_layer, {}, ".format(args.n_layers + 1)
                string += "n_hidden, {}, ".format(args.n_hidden)
                string += "best_acc, {:.3%}, ".format(np.max(test_accs))
                string += "acc_std, {:.3%}, ".format(np.std(test_accs))
                string += "mea_epoch_t, {:.3f}".format(np.mean(epoch_times))
                string += "epoch_t_std, {:.3f}".format(np.std(epoch_times))
                f.write(string + "\n")
    else:
        main(args, 0, name_base)

