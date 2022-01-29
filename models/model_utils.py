import torch
import os
import copy
import numpy as np
import dgl
from cache_sample import sample_rand_coo
import time

def drop_edge(g, sample_rate, device=None):
    if sample_rate < 1.0:
        adj = g.adj(scipy_fmt="coo")
        adj = sample_rand_coo(adj, sample_rate, verbose=False)
        g = dgl.from_scipy(adj, idtype=torch.int32, device=device)
        g = dgl.add_self_loop(g)
    return g

def save_model(path, model, fname):
    '''
    print("Model's state_dict info:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", 
                model.state_dict()[param_tensor].size())
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving model's state_dict as", fname)
    fname = os.path.join(path, fname)
    torch.save(model.state_dict(), fname)

def load_model(path, model, fname, gpu=0):
    print("Loading model's state_dict", path + '/' + fname)
    fname = os.path.join(path, fname)
    model.load_state_dict(torch.load(fname, map_location=f'cuda:{gpu}'))
    # model.load_state_dict(torch.load(fname))
    return model

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 99999999.9
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_model is None:
            self.best_score = val_loss
            self.best_model = copy.deepcopy(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_model = copy.deepcopy(model)
            self.counter = 0

    def get_best(self):
        return self.best_model

class BestVal:
    def __init__(self):
        self.val_loss_min = np.Inf
        self.best_model = None
        self.best_val_acc = -1

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.best_model = copy.deepcopy(model)

    # def __call__(self, val_acc, model):
    #     if val_acc > self.best_val_acc:
    #         self.best_val_acc = val_acc
    #         self.best_model = copy.deepcopy(model)

    def get_best(self):
        return self.best_model

class Log:
    def log_train(self, log_path, log_name, args, test_accs, epoch_times):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if args.early_stop is True:
            log_name = log_name + "_earlystop"
        elif args.best_val is True:
            log_name = log_name + "_bestval"

        log_name = log_name + "_log.csv"

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

    def log_prof_train(self, log_path, log_name, args, avg_epoch_t, std_epoch_t, 
                       avg_spmm_t, avg_mm_t, avg_sample_t=0):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        with open(log_name, 'a+') as f:
            string = "n_layer, {}, ".format(args.n_layers + 1)
            string += "n_hidden, {}, ".format(args.n_hidden)
            string += "S, {}, ".format(args.S)
            string += "sample_rate, {}, ".format(args.sr)
            string += "avg_epoch_t, {:.3f}, ".format(avg_epoch_t)
            string += "std_epoch_t, {:.3f}, ".format(std_epoch_t)
            string += "avg_spmm_t, {:.3f}, ".format(avg_spmm_t)
            string += "avg_mm_t, {:.3f}, ".format(avg_mm_t)
            string += "avg_sample_t, {:.3f}".format(avg_sample_t)
            f.write(string + "\n")

    def log_prof_infer(self, log_path, log_name, args, acc, avg_epoch_t, avg_spmm_t, 
                       avg_mm_t):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        with open(log_name, 'a+') as f:
            string = "n_layer, {}, ".format(args.n_layers + 1)
            string += "n_hidden, {}, ".format(args.n_hidden)
            string += "S, {}, ".format(args.S)
            string += "sample_rate, {}, ".format(args.sr)
            string += "acc, {:.3%}, ".format(acc)
            string += "avg_epoch_t, {:.3f}, ".format(avg_epoch_t)
            string += "avg_spmm_t, {:.3f}, ".format(avg_spmm_t)
            string += "avg_mm_t, {:.3f}, ".format(avg_mm_t)
            f.write(string + "\n")
