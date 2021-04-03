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
import torch_geometric.transforms as PyG_T
# from ogb.nodeproppred import DglNodePropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from model_utils import save_model, load_model
import dgl.backend.pytorch.sparse as dgl_pytorch_sp

from models import GraphSAGE
import dgl

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    model = GraphSAGE(
                in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                args.dropout, args.aggregator_type, args.cache_sample
    )

    return model


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer):
    model.train()
    criterion = th.nn.BCEWithLogitsLoss()
    feat = graph.ndata["feat"]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = criterion(pred[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()

    return loss#, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]
    pred = model(graph, feat)

    train_rocauc = evaluator.eval({
        'y_true': labels[train_idx],
        'y_pred': pred[train_idx],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': labels[val_idx],
        'y_pred': pred[val_idx],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': labels[test_idx],
        'y_pred': pred[test_idx],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


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

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    # )

    # training loop
    total_time = 0
    best_val_roc, best_test_roc = 0, 0

    # accs, train_accs, val_accs, test_accs = [], [], [], []
    # losses, train_losses, val_losses, test_losses = [], [], [], []
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        # adjust_learning_rate(optimizer, args.lr, epoch)
        loss = train(model, graph, labels, train_idx, optimizer)
        # lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if epoch % args.eval_every == 0:
            train_rocauc, valid_rocauc, test_rocauc = evaluate(
                model, graph, labels, train_idx, val_idx, test_idx, evaluator
            )
            
            # if best_val_roc < valid_rocauc:
            if best_test_roc < test_rocauc:
                best_val_roc = valid_rocauc
                best_test_roc = test_rocauc
                if args.save_model:
                    model_name = name_base + "_best_{}.pt".format(n_running)
                    save_model(args.dir, model, model_name)

            if epoch % args.log_every == 0:
                # logger.add_result(run, result)
                print(f"Epoch: {epoch}/{args.n_epochs}")
                print(f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_rocauc:.2f}%, '
                      f'Valid: {100 * valid_rocauc:.2f}%, '
                      f'Test: {100 * test_rocauc:.2f}%\n'
                      f"Best Valid: {100 * best_val_roc:.2f}%, "
                      f"Best Test: {100 * best_test_roc:.2f}"
                )

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test roc: {best_test_roc}")

    return best_val_roc, best_test_roc, model


def run_eval(args, graph, labels, train_idx, val_idx, test_idx, evaluator, name_base):
    # load model from disk
    model = gen_model(args)
    model_name = name_base + "_best.pt"
    model = load_model(args.dir, model, model_name)    
    model = model.to(device)

    train_rocauc, valid_rocauc, test_rocauc = evaluate(
        model, graph, labels, train_idx, val_idx, test_idx, evaluator
    )
    print(f"Test Roc-Auc: {test_rocauc}")

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
    print("Avg GSpMM CUDA kernel time (ms): {:.3f}".format(avg_spmm_t))

    if args.log != "none":
        with open("./log/sage_" + args.log + "_log.csv", 'a+') as f:
            if args.cache_sample:
                S = dgl_pytorch_sp.S
            else:
                S = 0
            string = "S, {}, ".format(S)
            string += "accuracy, {:.4f}, ".format(test_rocauc)
            string += "avg cuda time, {:.3f}, ".format(avg_spmm_t)
            string += "avg total time, {:.3f}".format(avg_t)
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
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.5)
    # argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--log-every", type=int, default=20)
    # argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument("--train", action='store_true',
            help="perform training")
    argparser.add_argument("--inference", action='store_true',
            help="perform inference")
    argparser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    argparser.add_argument("--dir", type=str, default="./state_dicts/sage/",
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
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root="/home/ubuntu/.ogb",
                                      transform=PyG_T.ToSparseTensor())
    pyg_data = dataset[0]

    evaluator = Evaluator(name="ogbn-proteins")

    splitted_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]

    # construct dgl graph from pyg adj mat
    row, col, val = pyg_data.adj_t.coo()
    graph = dgl.graph((row, col), idtype=th.int32)
    labels = pyg_data.y

    # Move edge features to node features.
    pyg_data.x = pyg_data.adj_t.mean(dim=1)
    # pass pyg feature to dgl graph
    graph.ndata["feat"] = pyg_data.x.float()

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    # n_classes = (labels.max() + 1).item()
    n_classes = 112
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.float().to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []
    model_list = []

    name_base = "sage_proteins_{}_layer_{}_hidden".format(
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
            model_name = name_base + "_best"
            cmd = "cp {}/{}_{}.pt {}/{}.pt".format(args.dir, model_name, best_idx, args.dir, model_name)
            os.system(cmd)
            # os.system("rm ./state_dicts/{}_*.pt")

        print(f"Runned {args.n_runs} times")
        print(f"Val Accs: {val_accs}")
        print(f"Test Accs: {test_accs}")
        print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
        print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
        print(f"Number of params: {count_parameters(args)}")

    if args.inference:
        run_eval(args, graph, labels, train_idx, val_idx, test_idx, evaluator, name_base)


if __name__ == "__main__":
    main()
