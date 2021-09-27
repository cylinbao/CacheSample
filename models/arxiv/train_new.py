#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
# from model_utils import save_model, load_model
from model_utils import save_model, load_model, EarlyStopping, BestVal, Log

from models import GCN, ResGCN, JKNet, GraphSAGE

# from cache_sample import sample_rand_coo
# import dgl.backend.pytorch.sparse as dgl_pytorch_sp

device = None
# in_feats, n_classes = None, None
# n_feats, n_classes = None, None
epsilon = 1 - math.log(2)

def gen_model(args, n_feats, n_classes):
    # if args.cache_sample:
    #     model = GCN(
    #                 in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, 
    #                 args.dropout, args.use_linear, norm="none"
    #             )
    # else:
    #     model = GCN(
    #                 in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, 
    #                 args.dropout, args.use_linear, norm="right"
    #             )

    if args.model == "gcn":
        model = GCN(n_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    args.dropout)
        # g = dgl.add_self_loop(g)
    elif args.model_type == "res": 
        model = ResGCN(in_dim=n_feats,
                       hid_dim=args.n_hidden,
                       out_dim=n_classes,
                       num_layers=args.n_layers,
                       dropout=args.dropout)
        # g = dgl.add_self_loop(g)
    elif args.model_type == "jkn":
        model = JKNet(in_dim=n_feats,
                      hid_dim=args.n_hidden,
                      out_dim=n_classes,
                      num_layers=args.n_layers,
                      dropout=args.dropout)
        # g = dgl.add_self_loop(g)
    elif args.model_type == "sage": 
        model = GraphSAGE(n_feats,
                          args.n_hidden,
                          n_classes,
                          args.n_layers,
                          args.dropout,
                          args.aggregator_type)

    return model

def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

# def train(model, graph, labels, train_idx, optimizer, use_labels):
#     model.train()
# 
#     feat = graph.ndata["feat"]
# 
#     if use_labels:
#         mask_rate = 0.5
#         mask = th.rand(train_idx.shape) < mask_rate
# 
#         train_labels_idx = train_idx[mask]
#         train_pred_idx = train_idx[~mask]
# 
#         feat = add_labels(feat, labels, train_labels_idx)
#     else:
#         mask_rate = 0.5
#         mask = th.rand(train_idx.shape) < mask_rate
# 
#         train_pred_idx = train_idx[mask]
# 
#     optimizer.zero_grad()
#     pred = model(graph, feat)
#     loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
#     loss.backward()
#     optimizer.step()
# 
#     return loss, pred

def train(model, graph, labels, train_idx, optimizer, args, norm_type):
    model.train()

    feat = graph.ndata["feat"]

    # mask_rate = 0.5
    # mask = th.rand(train_idx.shape) < mask_rate
    # train_pred_idx = train_idx[mask]
        
    t0 = time.time()
    seed = int((t0 - math.floor(t0))*1e7)

    optimizer.zero_grad()
    pred = model(graph, feat, norm_type=norm_type, kernel=args.kernel, 
                 S=args.S, seed=seed, sample_rate=args.sr)
    loss = cross_entropy(pred[train_idx], labels[train_idx])
    # loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred

@th.no_grad()
# def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
def evaluate_old(model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    # pred = model(graph, feat)
    pred = model(graph, feat, norm_type=norm_type, kernel=args.kernel, 
                 S=args.S, seed=seed, sample_rate=args.sr)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )

@th.no_grad()
def evaluate(model, graph, labels, mask_idx, evaluator, args, 
             norm_type="right", kernel="cuSPARSE", seed=0):
    model.eval()

    feat = graph.ndata["feat"]

    pred = model(graph, feat, norm_type=norm_type, kernel=kernel, 
                 S=args.S, seed=seed, sample_rate=args.sr)
    loss = cross_entropy(pred[mask_idx], labels[mask_idx])
    acc = compute_acc(pred[mask_idx], labels[mask_idx], evaluator) 

    return acc, loss

@th.no_grad()
def inference(model, graph):
    model.eval()
    tic = time.time()
    feat = graph.ndata["feat"]
    model(graph, feat)
    th.cuda.synchronize()

    return time.time() - tic

def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, 
        n_feats, n_classes, run_i, model_name):
    # define model and optimizer
    model = gen_model(args, n_feats, n_classes)
    model = model.to(device)

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    # )
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # training loop
    # total_time = 0
    # best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    # accs, train_accs, val_accs, test_accs = [], [], [], []
    # losses, train_losses, val_losses, test_losses = [], [], [], []

    model_name = model_name + "_{}.pt".format(run_i)
    if args.early_stop is True:
        early_stop = EarlyStopping(patience=args.patience)
    if args.best_val is True:
        best_val = BestVal()

    if args.kernel == "cuSPARSE":
        norm_type = 'right'
    else:
        norm_type = 'none'

    dur = []
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer, args, norm_type)
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        lr_scheduler.step(loss)

        toc = time.time()
        # total_time += toc - tic
        dur.append(toc - tic)

        val_acc, val_loss = evaluate(model, graph, labels, val_idx, evaluator, args,
                                     norm_type="right", kernel="cuSPARSE")

        if epoch % args.log_every == 0:
            print(f"Epoch: {epoch} | Time(ms): {dur[-1]*1000:.3f} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_acc:.4f}")

        if args.early_stop is True:
            early_stop(val_loss, model)
            if early_stop.early_stop:
                print("Early stopping.")
                model = early_stop.get_best()
                break

        if args.best_val is True:
            best_val(val_loss, model)
            # best_val(val_acc, model)

    if args.best_val is True:
        model = best_val.get_best()

    avg_epoch_t = np.mean(dur[3:])*1000
    test_acc, test_loss = evaluate(model, graph, labels, test_idx, evaluator, args,
                                   norm_type="right", kernel="cuSPARSE")

    print("*" * 50)
    print(f"Average epoch time: {avg_epoch_t:.4f}, Test acc: {test_acc:.3%}")

    if args.save_model:
        save_model(args.dir, model, model_name)

    with th.no_grad():
        th.cuda.empty_cache()

    return test_acc, avg_epoch_t 

def run_eval(args, graph, labels, train_idx, val_idx, test_idx, evaluator, name_base):
    # load model from disk
    model = gen_model(args)
    model_name = name_base + "_best.pt"
    model = load_model(args.dir, model, model_name)    
    model = model.to(device)

    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
        model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
    )
    print(f"Test acc: {test_acc}")

    infer_times = []
    import torch.autograd.profiler as profiler
    with profiler.profile(use_cuda=True) as prof:
        for i in range(args.n_runs):
            t = inference(model, graph)
            infer_times.append(t)
            print("Inference time (ms): {:.3f}".format(t*1000))
    avg_t = np.mean(infer_times[3:])*1000
    print("Average inference time: {:.3f}".format(avg_t))

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    events = prof.key_averages()
    for evt in events:
        if evt.key == "GSpMM":
            avg_spmm_t = evt.cuda_time*evt.count/args.n_runs/1000
        if evt.key == "matmul":
            avg_mtm_t = evt.cuda_time*evt.count/args.n_runs/1000
        if evt.key == "mm":
            avg_mm_t = evt.cuda_time*evt.count/args.n_runs/1000
    print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))
    print("Avg GEMM CUDA kernel time (ms): {:.3f}".format(avg_mtm_t+avg_mm_t))
    return

    if args.log != "none":
        with open("./log/gcn_" + args.log + "_log.csv", 'a+') as f:
            if args.cache_sample:
                S = dgl_pytorch_sp.S
            else:
                S = 0
            string = "S, {}, ".format(S)
            string += "accuracy, {:.4f}, ".format(test_acc)
            string += "avg cuda time, {:.3f}, ".format(avg_spmm_t)
            string += "avg total time, {:.3f}".format(avg_t)
            f.write(string + "\n")

def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser("GNN for OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--model", type=str, default="gcn", 
            help="model type, choose from [gcn, res, jkn, sage]")
    argparser.add_argument("--aggregator-type", type=str, default="mean",
            help="For GraphSAGE: Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument("--cpu", action="store_true", 
            help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=1)
    argparser.add_argument("--n-epochs", type=int, default=1000)
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--sr", type=float, default=1.0,
            help="edge sample rate")
    argparser.add_argument("--n-layers", type=int, default=1)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.75) # default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=0.0, # default=5e-4, 
            help="Weight for L2 Loss")
    argparser.add_argument("--log-every", type=int, default=20)
    # argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--train", action='store_true',
            help="perform training")
    argparser.add_argument("--prof-train", action='store_true',
            help="profile training time (default=False)")
    argparser.add_argument("--prof-infer", action='store_true',
            help="profile inference performance(default=False)")
    # argparser.add_argument("--inference", action='store_true',
    #         help="perform inference")
    # argparser.add_argument("--acc_analysis", action='store_true',
    #         help="perform inference")
    argparser.add_argument("--early-stop", action='store_true',
            help="whether to early stoearly stopp")
    argparser.add_argument("--patience", type=int, default=100,
            help="early stop patience")
    argparser.add_argument("--best-val", action='store_true',
            help="keep the best validation model")
    argparser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    argparser.add_argument("--dir", type=str, default="./state_dicts",
            help="directory to store model's state dict")
    argparser.add_argument("--log", action='store_true', help="log or not")
    argparser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Which SpMM kernel to use")
    argparser.add_argument("--S", type=int, default=0,
            help="Define S value for CacheSample kernel")
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    home = os.getenv("HOME")
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=home + "/.ogb")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph = graph.int()

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop()

    if args.model != "sage":
        graph = graph.add_self_loop()

    print(f"Total edges: {graph.number_of_edges()}")

    n_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # norm_type isn't valid for graphsage
    if args.kernel == "cuSPARSE":
        norm_type = 'right'
    else:
        norm_type = 'none'

    args.dataset = "arxiv"
    args.dir = args.dir + '/' + args.model + '/' + args.dataset 

    name_base = "{}_{}_layer_{}_hidden_{}_{}".format(args.model, args.dataset, 
                args.n_layers, args.n_hidden, args.kernel)
    model_name = name_base

    logger = Log()

    # run
    test_accs = []
    epoch_times = []
    if args.train:
        for i in range(args.n_runs):
            test_acc, epoch_t = run(args, graph, labels, train_idx, val_idx, 
                    test_idx, evaluator, n_feats, n_classes, i, model_name)
            test_accs.append(test_acc)
            epoch_times.append(epoch_t)

        if args.save_model:
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

        print(f"Runned {args.n_runs} times")
        print(f"Test Accs: {test_accs}")
        print(f"Best Test Accs: {np.max(test_accs):.3%}")
        print(f"Average Test accuracy: {np.mean(test_accs):.3%} ± {np.std(test_accs):.3%}")
        print(f"Mean Epoch Time: {np.mean(epoch_times):.3f} ± {np.std(epoch_times):.3f}")

        if args.log:
            log_path = "./train_log/{}".format(args.model)
            log_name = "{}/{}_{}_{}_train".format(log_path, args.model,
                    args.dataset, args.kernel)

            logger.log_train(log_path, log_name, args, test_accs, epoch_times)

if __name__ == "__main__":
    main()
