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
from ogb.nodeproppred import DglNodePropPredDataset

import dgl.nn.pytorch as dglnn
import dgl
import json

from tqdm import tqdm, trange

from model import GCN, SAGE

STATE_DICT_PATH = "learned_param.pt"

try:
    from dgl.backend.pytorch.sparse import S
    print(f"Running in customized kernel with S={S}")
except ImportError as e:
    S = None
    print("Running in default DGL")

def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = th.log(0.5 + y) - math.log(0.5)
    return th.mean(y)


def train(model, graph, labels, train_idx, optimizer, feat):
    model.train()

    mask = th.rand(train_idx.shape) < 0.5
    # mask = th.ones(train_idx.shape, dtype=th.bool)

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_idx[mask]], labels[train_idx[mask]])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, feat):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    graph : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_idx : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    """
    model.eval()

    pred = model(graph, feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx]),
        compute_acc(pred[val_idx], labels[val_idx]),
        compute_acc(pred[test_idx], labels[test_idx]),
        val_loss,
    )


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


@th.no_grad()
def evaluate_ez(model, labels, train_idx, val_idx, test_idx, pred):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    labels : The labels of all the nodes.
    val_idx : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    """
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx]),
        compute_acc(pred[val_idx], labels[val_idx]),
        compute_acc(pred[test_idx], labels[test_idx]),
        val_loss,
    )

def warmup_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        lr *= epoch / 50

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def inference(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat):
    norm = args.norm
    if args.sage:
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.agg_type, norm)
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, norm)
    model = model.to(device)
    model.load_state_dict(th.load(args.state_dict))
    model.eval()
    times = []
    if args.perf:
        prof = th.autograd.profiler.profile(use_cuda=True).__enter__()
    for run in tqdm(range(1, args.n_runs + 1)):
        tic = time.time()
        with th.no_grad():
            _ = model(graph, feat)
        toc = time.time()
        times.append(toc - tic)
    if args.perf:
        prof.__exit__(None, None, None)
        key_avg = prof.total_average()
        cpu = key_avg.self_cpu_time_total
        cuda = key_avg.cuda_time_total
        for evt in prof.key_averages():
            if evt.key == "GSpMM":
                kernel = evt.cuda_time*evt.count/args.n_runs
    with th.no_grad():
        _, val_acc, test_acc, _ = evaluate(model, graph, labels, train_idx, val_idx, test_idx, feat)
    return val_acc.item(), test_acc.item(), times, {"cuda": cuda, "cpu": cpu, "kernel": kernel} if prof else None


def run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat):
    # Define model and optimizer
    norm = args.norm
    if args.sage:
        model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.agg_type, norm)
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, norm)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=args.verbose)

    # Training loop
    total_time = 0
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float("inf")
    all_time = []

    with trange(1, args.n_epochs + 1) as t:
        if args.perf:
            prof = th.autograd.profiler.profile(use_cuda=True).__enter__()
        for epoch in t:
            tic = time.time()
            train_acc = 0

            t.set_postfix(best_val=best_val_acc, best_test=best_test_acc)

            warmup_learning_rate(optimizer, args.lr, epoch)

            loss, pred = train(model, graph, labels, train_idx, optimizer, feat)

            acc = compute_acc(pred[train_idx], labels[train_idx])

            train_acc, val_acc, test_acc, val_loss = evaluate_ez(model, labels, train_idx, val_idx, test_idx, pred)
            lr_scheduler.step(val_loss)

            toc = time.time()
            total_time += toc - tic
            all_time.append(toc - tic)
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_val_acc = val_acc.item()
                best_test_acc = test_acc.item()
            

            if args.log_every and epoch % args.log_every == 0:
                print(f"*** Epoch: {epoch} ***")
                print(
                    f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}\n"
                    f"Train/Val/Test: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}, Val Loss: {val_loss:.4f}"
                )
            

    print("******")
    print(f"Avg epoch time: {total_time / args.n_epochs}")
    if args.perf:
        prof.__exit__(None, None, None)
    
    if args.train:
        th.save(model.state_dict(), args.state_dict)

    train_acc, val_acc, test_acc, val_loss = evaluate(model, graph, labels, train_idx, val_idx, test_idx, feat)

    if val_loss < best_val_loss:
        best_val_loss = val_loss.item()
        best_val_acc = val_acc.item()
        best_test_acc = test_acc.item()
        
    return best_val_acc, best_test_acc, all_time, prof if args.perf else None


def main():
    argparser = argparse.ArgumentParser("OGBN-Arxiv")
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
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=0)
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
    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    labels = labels[:, 0]
    feat = graph.ndata["feat"]
    in_feats = feat.shape[1]

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    n_classes = (labels.max() + 1).item()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    feat = feat.to(device)

    # run
    cuda_time = []
    cpu_time = []

    if args.inference:
        stats = {
            "args": vars(args)
        }
        val_acc, test_acc, times, prof = inference(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat)
        if prof:
            cuda_time = prof["cuda"]
            cpu_time = prof["cpu"]
        print(f"Inference Finished. Val/test acc:{val_acc:.4f}/{test_acc:.4f}")
            
        stats[f"gnn"] = {
            "time": times,
            "val_accs": val_acc,
            "test_accs": test_acc,
            "cuda_time": prof["cuda"],
            "cpu_time": prof["cpu"],
            "kernel_time": prof["kernel"]
        }
        with open(f"{args.log_file}.json", "w") as file:
            file.write(json.dumps(stats, sort_keys=True, indent=4))
    else:
        val_acc, test_acc, times, prof = run(args, device, graph, in_feats, n_classes, labels, train_idx, val_idx, test_idx, feat)
        print(f"Train Finished. Best val/test acc:{val_acc:.4f}/{test_acc:.4f}")


if __name__ == "__main__":
    main()
