import argparse, time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
from profile import evaluate, prof_infer, prof_train
from model_utils import drop_edge, save_model, load_model, EarlyStopping, BestVal, Log

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from models import GCN, ResGCN, JKNet, GraphSAGE

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
    n_feats = features.shape[1]
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

    # norm_type isn't valid for graphsage
    if args.kernel == "cuSPARSE":
        norm_type = 'right'
    else:
        norm_type = 'none'

    # add self loop for the graph
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    # create GNN model
    if args.model == "gcn":
        model = GCN(n_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    args.dropout)
        g = dgl.add_self_loop(g)
    elif args.model == "res": 
        model = ResGCN(in_dim=n_feats,
                       hid_dim=args.n_hidden,
                       out_dim=n_classes,
                       num_layers=args.n_layers,
                       dropout=args.dropout)
        g = dgl.add_self_loop(g)
    elif args.model == "jkn":
        model = JKNet(in_dim=n_feats,
                      hid_dim=args.n_hidden,
                      out_dim=n_classes,
                      num_layers=args.n_layers,
                      dropout=args.dropout)
        g = dgl.add_self_loop(g)
    elif args.model == "sage": 
        model = GraphSAGE(n_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        args.dropout,
                        args.aggregator_type)
    else:
        raise ValueError('Unknown model type: {}'.format(args.model))

    if cuda:
        model.cuda()
    
    if args.prof_train is True:
        avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t = prof_train(args, model, g, features, train_mask, labels, norm_type)

        return avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t 
    elif args.prof_infer is True:
        acc, avg_t, avg_spmm_t, avg_mm_t = prof_infer(
                args, model_name, model, g, features, labels, test_mask, norm_type)
        return acc, avg_t, avg_spmm_t, avg_mm_t 

    # perform training
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # model_name = model_name + "_{}.pt".format(run_i)
    if args.early_stop is True:
        early_stop = EarlyStopping(patience=args.patience)
    if args.best_val is True:
        best_val = BestVal()

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()

        t0 = time.time()
        # seed = 0
        seed = int((t0 - math.floor(t0))*1e7)
        if args.drop_edge is True:
            _g = drop_edge(g, args.sr, device=features.get_device())
            logits = model(_g, features, norm_type=norm_type, kernel=args.kernel, 
                       S=args.S, seed=seed, sample_rate=args.sr)
        else:
            logits = model(g, features, norm_type=norm_type, kernel=args.kernel, 
                       S=args.S, seed=seed, sample_rate=args.sr)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # if epoch >= 3:
        dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, g, features, labels, val_mask, 
                                     norm_type='right', kernel='cuSPARSE')
        print("Epoch {:05d} | Time(ms) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Accuracy {:.4f} ".format(epoch, dur[-1]*1000, loss.item(), val_loss, val_acc))

        if args.early_stop is True:
            early_stop(val_loss, model)
            # early_stop(val_acc, model)
            if early_stop.early_stop:
                print("Early stopping.")
                model = early_stop.get_best()
                break

        if args.best_val is True:
            best_val(val_loss, model)
            # best_val(val_acc, model)

    if args.best_val is True:
        model = best_val.get_best()

    print()
    val_loss, val_acc = evaluate(model, g, features, labels, val_mask, 
                                 norm_type='right', kernel='cuSPARSE')
    print("Val Accuracy {:.5f}".format(val_acc))
    test_loss, test_acc = evaluate(model, g, features, labels, test_mask, 
                                   norm_type='right', kernel='cuSPARSE')
    print("Test Accuracy {:.5f}".format(test_acc))

    epoch_t = np.mean(dur[3:])*1000
    print("Total epoch time (s): {:.3f}".format(np.sum(dur)))
    print("Mean epoch time (ms): {:.3f}".format(epoch_t))

    # if args.save_model or args.best_val or args.early_stop:
    if args.save_model: 
        save_model(args.dir, model, model_name)

    with torch.no_grad():
        torch.cuda.empty_cache()

    return test_acc, epoch_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str, default="gcn", 
            help="model type, choose from [gcn, res, jkn, sage]")
    parser.add_argument("--aggregator-type", type=str, default="mean",
            help="For GraphSAGE: Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--drop-edge", action='store_true',
            help="drop edge in python")
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
    parser.add_argument("--train", action='store_true',
            help="perform training")
    parser.add_argument("--prof-train", action='store_true',
            help="profile training time (default=False)")
    parser.add_argument("--prof-infer", action='store_true',
            help="profile inference performance(default=False)")
    parser.add_argument("--dir", type=str, default="./state_dicts",
            help="directory to store model's state dict")
    parser.add_argument("--save-model", action='store_true',
            help="whether to save model")
    parser.add_argument("--early-stop", action='store_true',
            help="whether to early stoearly stopp")
    parser.add_argument("--patience", type=int, default=100,
            help="early stop patience")
    parser.add_argument("--best-val", action='store_true',
            help="keep the best validation model")
    parser.add_argument("--log", action='store_true', help="log or not")
    parser.add_argument("--kernel", type=str, default="cuSPARSE",
            help="Which SpMM kernel to use")
    parser.add_argument("--S", type=int, default=0,
            help="Define S value for CacheSample kernel")
    args = parser.parse_args()

    args.dir = args.dir + '/' + args.model + '/' + args.dataset
    print(args)

    name_base = "{}_{}_layer_{}_hidden_{}_{}".format(args.model, args.dataset, 
                args.n_layers, args.n_hidden, args.kernel)
    # name_base = "{}_{}_layer_{}_hidden_{}_{}".format(args.model,
    #              args.dataset, args.n_layers, args.n_hidden, "cuSPARSE")

    if args.drop_edge is True:
        name_base += "_dropedge_sr_{:2d}".format(int(args.sr*100))

    model_name = name_base
    if "CacheSample1" in args.kernel:
        model_name = model_name + "_S_{}".format(args.S)
    elif "CacheSample2" in args.kernel:
        model_name = model_name + "_sr_{}".format(args.sr)

    assert args.train ^ args.prof_train ^ args.prof_infer, "only one mode is allowed"

    logger = Log()

    if args.train:
        if args.kernel == "cuSPARSE":
            lr_rates = [0.01]
        else:
            lr_rates = np.arange(0.001, 0.011, 0.001)

        for lr in lr_rates:
            args.lr = lr
            test_accs = []
            epoch_times = []
            for run_i in range(args.n_runs):
                _model_name = model_name + "_run_{}.pt".format(run_i)
                test_acc, epoch_t = run(args, run_i, _model_name)
                test_accs.append(test_acc)
                epoch_times.append(epoch_t)

            print()
            print(f"Test Accs: {test_accs}")
            print(f"Best Test Accuracy: {np.max(test_accs):.3%}")
            print(f"Average Test accuracy: {np.mean(test_accs):.3%} ± {np.std(test_accs):.3%}")
            print(f"Mean Epoch Time: {np.mean(epoch_times):.3f} ± {np.std(epoch_times):.3f}")

            if args.save_model:
                best_idx = np.argmax(test_accs)
                print("best_idx: ", best_idx)
                cmd = "cp {}/{}_run_{}.pt {}/{}_best.pt".format(args.dir, model_name, 
                        best_idx, args.dir, model_name)
                print(cmd)
                os.system(cmd)
                for i in range(args.n_runs):
                    cmd = "rm {}/{}_run_{}.pt".format(args.dir, model_name, i)
                    print(cmd)
                    os.system(cmd)

            if args.log:
                log_path = "./train_log/{}".format(args.model)
                log_name = "{}/{}_{}_{}_train".format(log_path, args.model,
                            args.dataset, args.kernel)
                if args.drop_edge is True:
                    log_name += "_dropedge"

                logger.log_train(log_path, log_name, args, test_accs)
    elif args.prof_train:
        # avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t = run(args, 0, model_name)
        # if args.log:
        #     log_path = "./prof_train/{}".format(args.model)
        #     log_name = "{}/{}_{}_{}".format(log_path, args.model, args.dataset, 
        #                                     args.kernel)
        #     if args.drop_edge is True:
        #         log_name += "_dropedge"
        #     log_name += "_prof_train_log.csv"

        #     logger.log_prof_train(log_path, log_name, args, avg_epoch_t, std_epoch_t, 
        #                           avg_spmm_t, avg_mm_t, avg_sample_t)
        
        if args.kernel == "cuSPARSE" and args.drop_edge is False:
            avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t = run(args, 0, model_name)
            if args.log:
                log_path = "./prof_train/{}".format(args.model)
                log_name = "{}/{}_{}_{}".format(log_path, args.model, args.dataset, 
                                                args.kernel)
                log_name += "_prof_train_log.csv"

                logger.log_prof_train(log_path, log_name, args, avg_epoch_t, std_epoch_t, 
                                      avg_spmm_t, avg_mm_t, avg_sample_t)
        else:
            # rates = [0.3, 0.5, 0.7]
            rates = [0.5]
            for sr in rates:
                args.sr = sr
                avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t = run(args, 0, model_name)
                if args.log:
                    log_path = "./prof_train/{}".format(args.model)
                    log_name = "{}/{}_{}_{}".format(log_path, args.model, args.dataset, 
                                                    args.kernel)
                    if args.drop_edge is True:
                        log_name += "_dropedge"
                    log_name += "_prof_train_log.csv"

                    logger.log_prof_train(log_path, log_name, args, avg_epoch_t, 
                                    std_epoch_t, avg_spmm_t, avg_mm_t, avg_sample_t)
    elif args.prof_infer:
        if args.kernel == "cuSPARSE":
            acc, avg_epoch_t, avg_spmm_t, avg_mm_t = run(args, 0, model_name)

            if args.log:
                log_path = "./prof_infer/{}".format(args.model)
                log_name = "{}/{}_{}_{}_infer_log.csv".format(log_path, args.model,
                            args.dataset, args.kernel)

                logger.log_prof_infer(log_path, log_name, args, acc, avg_epoch_t, avg_spmm_t, avg_mm_t)
        else:
            rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            spmm_t = []
            for sr in rates:
                args.sr = sr
                acc, avg_epoch_t, avg_spmm_t, avg_mm_t = run(args, 0, model_name)
                spmm_t.append(avg_spmm_t)

                if args.log:
                    log_path = "./prof_infer/{}".format(args.model)
                    log_name = "{}/{}_{}_{}_infer_log.csv".format(log_path, args.model,
                                args.dataset, args.kernel)

                    logger.log_prof_infer(log_path, log_name, args, acc, avg_epoch_t, avg_spmm_t, avg_mm_t)

            print(spmm_t)
