import torch
import os
import copy
import numpy as np

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
    return model

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 99999999.9
        self.early_stop = False
        self.best_model = None

    # def __call__(self, val_acc, model):
    #     if self.best_model is None:
    #         self.best_score = val_acc
    #         self.best_model = copy.deepcopy(model)
    #     elif val_acc < self.best_score:
    #         self.counter += 1
    #         if self.verbose:
    #             print("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = val_acc
    #         self.best_model = copy.deepcopy(model)
    #         self.counter = 0

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
    # def __init__(self):
    #     self.val_acc_max = 0
    #     self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.best_model = copy.deepcopy(model)
    # def __call__(self, val_acc, model):
    #     if val_acc > self.val_acc_max:
    #         self.val_acc_max = val_acc
    #         self.best_model = copy.deepcopy(model)

    def get_best(self):
        return self.best_model
