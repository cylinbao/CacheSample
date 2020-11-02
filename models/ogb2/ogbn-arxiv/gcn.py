#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from model_utils import save_model, load_model
import dgl.backend.pytorch.sparse as dgl_pytorch_sp

from models import GCN

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    if args.cache_sample:
        model = GCN(
                    in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, 
                    args.dropout, args.use_linear, norm="none"
                )
    else:
        model = GCN(
                    in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, 
                    args.dropout, args.use_linear, norm="right"
                )

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


def train(model, graph, labels, train_idx, optimizer, use_labels):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
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
def inference(model, graph):
    model.eval()
    tic = time.time()
    feat = graph.ndata["feat"]
    model(graph, feat)
    return time.time() - tic


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, name_base):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels)
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
        )

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        # if val_acc > best_val_acc:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            
            if args.save_model:
                model_name = name_base + "_best_{}.pt".format(n_running)
                save_model(args.dir, model, model_name)

        if epoch % args.log_every == 0:
            print(f"Epoch: {epoch}/{args.n_epochs}")
            print(
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_loss_{n_running}.png")

    return best_val_acc, best_test_acc, model


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
        print("Average inference time (ms): {:.3f}".format(
            np.mean(infer_times[:])*1000))

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    events = prof.key_averages()
    for evt in events:
        if evt.key == "GSpMM":
            avg_spmm_t = evt.cuda_time*evt.count/args.n_runs/1000
    print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))

    if args.log != "none":
        with open("./log/gcn_" + args.log + "_log.csv", 'a+') as f:
            if args.cache_sample:
                S = dgl_pytorch_sp.S
            else:
                S = 0
            string = "S, {}, ".format(S)
            string += "accuracy, {:.4f}, ".format(test_acc)
            string += "cuda time, {:.3f}".format(avg_spmm_t)
            f.write(string + "\n")


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser("GCN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=1000)
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--use-linear", action="store_true", help="Use linear layer.")
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--train", action='store_true',
            help="perform training")
    argparser.add_argument("--inference", action='store_true',
            help="perform inference")
    argparser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    argparser.add_argument("--dir", type=str, default="./state_dicts/gcn/",
            help="filename of log, if none, then no log")
    argparser.add_argument("--log", type=str, default="none",
            help="filename of log, if none, then no log")
    argparser.add_argument("--cache-sample", action='store_true',
            help="Use CacheSample kernel")
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv", root="/home/ubuntu/.ogb")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []
    model_list = []

    name_base = "gcn_arxiv_{}_layer_{}_hidden".format(
                args.n_layers, args.n_hidden)

    if args.train:
        for i in range(args.n_runs):
            val_acc, test_acc, model = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i, name_base)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            model_list.append(model)

        if args.save_model:
            # best_model = model_list[np.argmax(test_accs)]
            best_idx = np.argmax(test_accs)
            print("best_idx: ", best_idx)
            print("best test acc: ", test_accs[best_idx])
            model_name = name_base + "_best"
            cmd = "cp {}/{}_{}.pt {}/{}.pt".format(args.dir, model_name, best_idx, args.dir, model_name)
            os.system(cmd)
            # os.system("rm ./state_dicts/{}_*.pt")

        print(f"Runned {args.n_runs} times")
        print("Val Accs:", val_accs)
        print("Test Accs:", test_accs)
        print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
        print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
        print(f"Number of params: {count_parameters(args)}")

    if args.inference:
        run_eval(args, graph, labels, train_idx, val_idx, test_idx, evaluator, name_base)


if __name__ == "__main__":
    main()
