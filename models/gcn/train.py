import argparse, time
import numpy as np
import networkx as nx
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model_utils import save_model, load_model

from gcn import GCN

def evaluate(model, g, features, labels, mask, norm='right', norm_bias=0, 
        kernel='cuSPARSE', S=0, seed=0):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, norm, norm_bias, kernel, S, seed)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# Run forward and return runtime
def inference(model, g, features, norm='right', norm_bias=0, 
        kernel='cuSPARSE', S=0, seed=0):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(g, features, norm, norm_bias, kernel, S, seed)
    torch.cuda.synchronize()

    return time.time() - t0

def run(args, n_run, name_base):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
    	data = RedditDataset(self_loop=True)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
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

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    if args.kernel == "cuSPARSE":
        norm = 'right'
    else:
        norm = 'none'

    # create GCN model
    model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    
    if args.inference:
        model_name = name_base + "_best.pt"
        model = load_model(args.dir, model, model_name)
        acc = evaluate(model, g, features, labels, test_mask, 
                norm, args.norm_bias, args.kernel, args.S)
        # acc = evaluate(model, g, features, labels, test_mask, 'right', 0, 'cuSPARSE')
        print("Test accuracy {:.3%}".format(acc))

        num_run = 10
        times = []

        with profiler.profile(use_cuda=True) as prof:
            for i in range(num_run):
                t = inference(model, g, features, norm, args.norm_bias, 
                        args.kernel, args.S)
                times.append(t)
                print("Inference time: {:.3f}".format(t))
        avg_t = np.mean(times[3:])*1000
        print("Average inference time: {:.3f}".format(avg_t))

        # print(prof.key_averages().table(sort_by="cuda_time_total"))

        events = prof.key_averages()
        for evt in events:
            if evt.key == "GSpMM":
                # print(evt.self_cuda_time_total_str)
                avg_spmm_t = evt.cuda_time*evt.count/num_run/1000
            # if evt.key == "matmul":
            #     avg_matmul_t = evt.cuda_time*evt.count/num_run/1000
            # if evt.key == "mm":
            #     avg_mm_t = evt.cuda_time*evt.count/num_run/1000
        print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))
        # print("Avg GEMM CUDA kernel time (ms): {:.3f}".format(avg_mm_t+avg_matmul_t))

        if args.log != "none":
            with open("./log/" + args.dataset + "_" + args.log + "_log.csv", 'a+') as f:
                if args.cache_sample:
                    S = dgl_pytorch_sp.S
                else:
                    S = 0
                string = "S, {}, ".format(S)
                string += "accuracy, {:.4f}, ".format(acc)
                string += "avg cuda time, {:.3f}, ".format(avg_spmm_t)
                string += "avg total time, {:.3f}".format(avg_t)
                f.write(string + "\n")
        return 

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    # with profiler.profile(use_cuda=True) as prof:
    for epoch in range(args.n_epochs):
        model.train()

        t0 = time.time()
        seed = int((t0 - math.floor(t0))*1e7)
        # forward
        logits = model(g, features, norm=norm, norm_bias=args.norm_bias, 
                        kernel=args.kernel, S=args.S, seed=seed)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch >= 3:
        dur.append(time.time() - t0)

        val_acc = evaluate(model, g, features, labels, val_mask, 'right', 0, 'cuSPARSE',  0, 0)
        print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, dur[-1]*1000, 
            loss.item(), val_acc))
        # print("Epoch {:05d} | Time(ms) {:.4f}".format(epoch, dur[-1]*1000))
        # format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))
                                        
        # if best_val_acc < acc:
        #     best_val_acc = acc
        #     if args.save_model:
        #         model_name = name_base + "_{}.pt".format(n_run)
        #         save_model(args.dir, model, model_name)

    # print(prof.key_averages().table(sort_by="cuda_time_total"))

    epoch_t = np.mean(dur[3:])*1000

    print()
    print("Total epoch time (ms): {:.3f}".format(np.sum(dur)))
    print("Mean epoch time (ms): {:.3f}".format(epoch_t))
    # acc = evaluate(model, g, features, labels, test_mask)
    test_acc = evaluate(model, g, features, labels, test_mask, 'right', 0, 'cuSPARSE', 0, 0)
    print("Test accuracy {:.2%}".format(test_acc))

    if args.save_model:
        model_name = name_base + "_{}.pt".format(n_run)
        save_model(args.dir, model, model_name)

    with torch.no_grad():
        torch.cuda.empty_cache()

    return test_acc, epoch_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-runs", type=int, default=1,
            help="number of training runs")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    #parser.add_argument("--norm", type=str, default='both',
    #        help="setup norm strategy")
    parser.add_argument("--train", action='store_true',
            help="perform training")
    parser.add_argument("--inference", action='store_true',
            help="whether to just perform inference (default=False)")
    parser.add_argument("--dir", type=str, default="./state_dicts",
            help="directory to store model's state dict")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    # parser.add_argument("--log", type=str, default="none",
    #         help="filename of log, if none, then no log")
    parser.add_argument("--log", action='store_true', help="log or not")
    parser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Define kernel from cuSPARSE and CacheSample")
    parser.add_argument("--norm-bias", type=int, default=0,
            help="Define norm bias for CacheSample kernel")
    parser.add_argument("--S", type=int, default=0,
            help="Define S value for CacheSample kernel")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    midle_dir = "{}_S{}".format(args.kernel, args.S)
    # midle_dir = "cuSPARSE_S0"
    args.dir = args.dir + '/' + args.dataset + '/' + midle_dir 
    print(args)

    name_base = "gcn_{}_{}_layer_{}_hidden".format(
                 args.dataset, args.n_layers, args.n_hidden)

    test_accs = []
    epoch_times = []
    if args.train:
        for i in range(args.n_runs):
            test_acc, epoch_t = run(args, i, name_base)
            test_accs.append(test_acc)
            epoch_times.append(epoch_t)

        if args.save_model:
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

        print()
        print(f"Runned {args.n_runs} times")
        print(f"Test Accs: {test_accs}")
        print(f"Best Test Accuracy: {np.max(test_accs):.3%}")
        print(f"Average Test accuracy: {np.mean(test_accs):.3%} ± {np.std(test_accs):.3%}")
        print(f"Mean Epoch Time: {np.mean(epoch_times):.3f}")

        if args.log:
            with open("./log/{}/gcn_{}_{}_S{}_train_log.csv".format(
                    args.dataset, args.dataset, args.kernel, args.S), 'a+') as f:
                string = "n_layer, {}, ".format(args.n_layers + 1)
                string += "n_hidden, {}, ".format(args.n_hidden)
                string += "best_acc, {:.3%}, ".format(np.max(test_acc))
                string += "mea_epoch_t, {:.3f}".format(np.mean(epoch_times))
                f.write(string + "\n")

    if args.inference:
        run(args, 0, name_base)
