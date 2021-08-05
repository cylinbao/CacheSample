import torch
import os
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
    def __init__(self, patience=10, path=None, fname=None, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.fname = fname

        # if path is not None and fname is not None:
        #     self.fname = os.path.join(path, fname)
        # else:
        #     print("Please Fix path and fname for early stopping")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f --> %.6f).  Saving model ...'%(self.val_loss_min, val_loss))
        # torch.save(model.state_dict(), self.fname)
        save_model(self.path, model, self.fname)
        self.val_loss_min = val_loss

    def load_checkpoint(self, gpu=0):
        fname = os.path.join(self.path, self.fname)
        return torch.load(fname, map_location=f'cuda:{gpu}')
