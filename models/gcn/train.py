import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model_utils import save_model, load_model
from cache_sample import sample_rand_coo
import dgl.backend.pytorch.sparse as dgl_pytorch_sp

from gcn import GCN

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# Run forward and return runtime
def inference(model, g, features):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(g, features)
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

    # if args.cache_sample:
    #     kernel = "CacheSample"
    #     norm = 'none'
    # else:
    #     kernel = "cuSPARSE"
    #     norm = 'right'

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
                args.dropout,
                norm,
                args.kernel,
                args.S)

    if cuda:
        model.cuda()
    
    if args.inference:
        model_name = name_base + "_best.pt"
        model = load_model(args.dir, model, model_name)
        acc = evaluate(model, g, features, labels, test_mask)
        print("Test accuracy {:.3%}".format(acc))

        num_run = 10
        times = []

        import torch.autograd.profiler as profiler
        with profiler.profile(use_cuda=True) as prof:
            for i in range(num_run):
                t = inference(model, g, features)
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
            if evt.key == "matmul":
                avg_matmul_t = evt.cuda_time*evt.count/num_run/1000
            if evt.key == "mm":
                avg_mm_t = evt.cuda_time*evt.count/num_run/1000
        print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))
        print("Avg GEMM CUDA kernel time (ms): {:.3f}".format(avg_mm_t+avg_matmul_t))
        return

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
    best_val_acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
                                    
        if best_val_acc < acc:
            best_val_acc = acc
            if args.save_model:
                model_name = name_base + "_best_{}.pt".format(n_run)
                save_model(args.dir, model, model_name)

    print()
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-runs", type=int, default=10,
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
    parser.add_argument("--log", type=str, default="none",
            help="filename of log, if none, then no log")
    parser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Define kernel from cuSPARSE and CacheSample")
    parser.add_argument("--S", type=int, default=128,
            help="Define S value for CacheSample kernel")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.dir = args.dir + '/' + args.dataset 
    print(args)

    # main(args)

    test_accs = []
    name_base = "gcn_{}_{}_layer_{}_hidden".format(
                 args.dataset, args.n_layers, args.n_hidden)
    if args.train:
        for i in range(args.n_runs):
            test_acc = run(args, i, name_base)
            test_accs.append(test_acc)

        if args.save_model:
            best_idx = np.argmax(test_accs)
            print("best_idx: ", best_idx)
            model_name = name_base + "_best"
            cmd = "cp {}/{}_{}.pt {}/{}.pt".format(args.dir, model_name, best_idx, args.dir, model_name)
            os.system(cmd)
            # os.system("rm ./state_dicts/{}_*.pt")

        print(f"Runned {args.n_runs} times")
        print(f"Test Accs: {test_accs}")
        print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")

    if args.inference:
        run(args, 0, name_base)
