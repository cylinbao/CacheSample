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
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from model_utils import save_model, load_model, EarlyStopping, BestVal

from sage import GraphSAGE

def evaluate(model, g, features, labels, mask, norm_type='right', 
             norm_bias=0, kernel='cuSPARSE', S=0, seed=0):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, norm_type, norm_bias, kernel, S, seed)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        loss = F.cross_entropy(logits, labels)
        print("{:.4f}".format(acc))
        return loss.item(), acc

# Run forward and return runtime
def inference(model, g, features, norm_type='right', 
              norm_bias=0, kernel='cuSPARSE', S=0, seed=0):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(g, features, norm_type, norm_bias, kernel, S, seed)
    torch.cuda.synchronize()

    return time.time() - t0

def prof_infer(args, name_base, model, g, features, labels, test_mask, norm_type):
    model_name = name_base + "_best.pt"
    model = load_model(args.dir, model, model_name)

    accs = []
    for i in range(args.n_runs):
        t0 = time.time()
        seed = int((t0 - math.floor(t0))*1e7)
        loss, acc = evaluate(model, g, features, labels, test_mask, 
                 norm_type, args.norm_bias, args.kernel, args.S, seed)
        print("Test accuracy {:.3%}".format(acc))
        accs.append(acc)
    print()
    max_acc = np.max(accs)
    avg_acc = np.mean(accs)
    print("Max Accuracy: {:.3%}".format(max_acc))
    print("Avg Accuracy: {:.3%}".format(avg_acc))

    args.n_runs = 25
    times = []
    with profiler.profile(use_cuda=True) as prof:
        for i in range(args.n_runs):
            t = inference(model, g, features, norm_type, args.norm_bias, 
                    args.kernel, args.S)
            times.append(t)
            print("Inference time: {:.3f}".format(t))
    avg_t = np.mean(times[3:])*1000
    print()
    print("Average inference time: {:.3f}".format(avg_t))

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    events = prof.key_averages()
    avg_mm_t = 0
    for evt in events:
        if evt.key == "GSpMM":
            avg_spmm_t = evt.cuda_time*evt.count/args.n_runs/1000
        if evt.key == "aten::matmul":
            avg_mm_t += evt.cuda_time*evt.count/args.n_runs/1000
        if evt.key == "aten::mm":
            avg_mm_t += evt.cuda_time*evt.count/args.n_runs/1000

    print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))
    print("Avg GEMM CUDA kernel time (ms): {:.3f}".format(avg_mm_t))

    return max_acc, avg_acc, avg_t, avg_spmm_t, avg_mm_t 

def run(args, run_i, model_name):
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
        norm_type = 'right'
    else:
        norm_type = 'none'

    # create model
    # model = GCN(in_feats,
    #             args.n_hidden,
    #             n_classes,
    #             args.n_layers,
    #             F.relu,
    #             args.dropout)
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()
    
    if args.prof_infer:
        max_acc, avg_acc, avg_t, avg_spmm_t, avg_mm_t = prof_infer(
                args, model_name, model, g, features, labels, test_mask, norm_type)
        return max_acc, avg_acc, avg_t, avg_spmm_t, avg_mm_t 

    # perform training
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    model_name = model_name + "_{}.pt".format(run_i)
    if args.early_stop is True:
        early_stop = EarlyStopping(path=args.dir, fname=model_name, verbose=False)
    if args.best_val is True:
        best_val = BestVal(path=args.dir, fname=model_name)

    # initialize graph
    dur = []
    # with profiler.profile(use_cuda=True) as prof:
    for epoch in range(args.n_epochs):
        model.train()

        t0 = time.time()
        seed = int((t0 - math.floor(t0))*1e7)
        # forward
        logits = model(g, features, norm_type=norm_type, norm_bias=args.norm_bias, 
                    kernel=args.kernel, S=args.S, seed=seed, sample_rate=args.sr)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # if epoch >= 3:
        dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, g, features, labels, val_mask, 'right', 0, 'cuSPARSE',  0, 0)
        print("Epoch {:05d} | Time(ms) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Accuracy {:.4f} ".format(epoch, dur[-1]*1000, loss.item(), val_loss, val_acc))

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
    val_loss, val_acc = evaluate(model, g, features, labels, val_mask, 'right', 0, 'cuSPARSE', 0, 0)
    print("Val Accuracy {:.5f}".format(val_acc))
    test_loss, test_acc = evaluate(model, g, features, labels, test_mask, 'right', 0, 'cuSPARSE', 0, 0)
    print("Test Accuracy {:.5f}".format(test_acc))

    epoch_t = np.mean(dur[3:])*1000
    print("Total epoch time (ms): {:.3f}".format(np.sum(dur)))
    print("Mean epoch time (ms): {:.3f}".format(epoch_t))

    if args.save_model:
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
    parser.add_argument("--sr", type=float, default=1.0,
            help="edge sample rate")
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
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    #parser.add_argument("--norm", type=str, default='both',
    #        help="setup norm strategy")
    parser.add_argument("--train", action='store_true',
            help="perform training")
    parser.add_argument("--prof-infer", action='store_true',
            help="profile inference performance(default=False)")
    parser.add_argument("--dir", type=str, default="./state_dicts",
            help="directory to store model's state dict")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    parser.add_argument("--early-stop", action='store_true',
            help="whether to early stoearly stopp")
    parser.add_argument("--best-val", action='store_true',
            help="keep the best validation model")
    parser.add_argument("--log", action='store_true', help="log or not")
    parser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Define kernel from cuSPARSE and CacheSample")
    parser.add_argument("--norm-bias", type=int, default=0,
            help="Define norm bias for CacheSample kernel")
    parser.add_argument("--S", type=int, default=0,
            help="Define S value for CacheSample kernel")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.dir = args.dir + '/' + args.dataset + '/' + args.kernel 
    print(args)

    name_base = "sage_{}_{}_agg_{}_layer_{}_hidden".format(args.dataset, 
                args.aggregator_type, args.n_layers, args.n_hidden)
    model_name = name_base
    if "CacheSample1" in args.kernel:
        model_name = model_name + "_{}_S".format(args.S)
    elif "CacheSample2" in args.kernel:
        model_name = model_name + "_{}_sr".format(args.sr)

    test_accs = []
    epoch_times = []
    if args.train:
        for i in range(args.n_runs):
            test_acc, epoch_t = run(args, i, model_name)
            test_accs.append(test_acc)
            epoch_times.append(epoch_t)

        print()
        print(f"Test Accs: {test_accs}")
        print(f"Best Test Accuracy: {np.max(test_accs):.3%}")
        print(f"Average Test accuracy: {np.mean(test_accs):.3%} ± {np.std(test_accs):.3%}")
        print(f"Mean Epoch Time: {np.mean(epoch_times):.3f} ± {np.std(epoch_times):.3f}")

        if args.save_model or args.early_stop or args.best_val:
            best_idx = np.argmax(test_accs)
            print("best_idx: ", best_idx)
            cmd = "cp {}/{}_{}.pt {}/{}_best.pt".format(args.dir, model_name, 
                    best_idx, args.dir, model_name)
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
            log_name = "{}/sage_{}_{}_agg_{}_train_log.csv".format(log_path, 
                        args.dataset, args.aggregator_type, args.kernel)
            with open(log_name, 'a+') as f:
                string = "n_layer, {}, ".format(args.n_layers + 1)
                string += "n_hidden, {}, ".format(args.n_hidden)
                string += "S, {}, ".format(args.S)
                string += "sample_rate, {}, ".format(args.sr)
                string += "best_acc, {:.3%}, ".format(np.max(test_accs))
                string += "acc_std, {:.3%}, ".format(np.std(test_accs))
                string += "mean_epoch_t, {:.3f}, ".format(np.mean(epoch_times))
                string += "epoch_t_std, {:.3f}".format(np.std(epoch_times))
                f.write(string + "\n")

    if args.prof_infer:
        max_acc, avg_acc, avg_t, avg_spmm_t, avg_mm_t = run(args, 0, model_name)

        if args.log:
            log_path = "./log/{}".format(args.dataset)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_name = "{}/sage_{}_{}_agg_{}_infer_log.csv".format(log_path, 
                        args.dataset, args.aggregator_type, args.kernel)
            with open(log_name, 'a+') as f:
                string = "n_layer, {}, ".format(args.n_layers + 1)
                string += "n_hidden, {}, ".format(args.n_hidden)
                string += "S, {}, ".format(args.S)
                string += "sample_rate, {}, ".format(args.sr)
                string += "max_acc, {:.3%}, ".format(max_acc)
                string += "avg_acc, {:.3%}, ".format(avg_acc)
                string += "avg_epoch_t, {:.3f}, ".format(avg_t)
                string += "avg_spmm_t, {:.3f}, ".format(avg_spmm_t)
                string += "avg_mm_t, {:.3f}, ".format(avg_mm_t)
                f.write(string + "\n")
