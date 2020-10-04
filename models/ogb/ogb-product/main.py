import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
import json

from load_graph import load_reddit, load_ogb, inductive_split

STATE_DICT_PATH = "learned_param.pt"
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        kernel = 0
        for l, layer in enumerate(self.layers):
            print(f"inferecing at layer: {l}")
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size * 2,
                shuffle=True,
                drop_last=False,
                num_workers=0 if args.perf else args.num_workers)

            if args.perf:
                prof = th.autograd.profiler.profile(use_cuda=True).__enter__()
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            if args.perf:
                prof.__exit__(None, None, None)
                key_avg = prof.total_average()
                cpu = key_avg.self_cpu_time_total
                cuda = key_avg.cuda_time_total
                for evt in prof.key_averages():
                    if evt.key == "GSpMM":
                        kernel += evt.cuda_time*evt.count

            x = y
        return y, kernel

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred, _ = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    if args.train:
        avg = 0
        iter_tput = []
        for epoch in range(args.num_epochs):
            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            tic_step = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # Load the input features as well as output labels
                #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]
                batch_inputs = blocks[0].srcdata['features']
                batch_labels = blocks[-1].dstdata['labels']

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                tic_step = time.time()

            toc = time.time()
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                eval_acc = evaluate(model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.batch_size, device)
                print('Eval Acc {:.4f}'.format(eval_acc))
                test_acc = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.batch_size, device)
                print('Test Acc: {:.4f}'.format(test_acc))
        if args.state_dict:
            th.save(model.state_dict(), args.state_dict)
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    elif args.inference:
        model.load_state_dict(th.load(args.state_dict))
        model.eval()
        times = []
        kernel_all = 0
        for _run in tqdm.tqdm(range(args.num_runs)):
            tic = time.time()
            with th.no_grad():
                pred, kernel = model.inference(test_g, test_g.ndata['features'], args.batch_size, device)
            times.append(time.time() - tic)
            kernel_all += kernel
        test_acc = compute_acc(pred, test_g.ndata['labels'])
        stats = {
            "args": vars(args),
            "gnn": {
                "time": times,
                "test_accs": test_acc.item(),
                "kernel_time": kernel/args.num_runs
            }
        }
        with open(f"{args.log_file}.json", "w") as file:
            file.write(json.dumps(stats, sort_keys=True, indent=4))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-runs", type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='15, 10, 5')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--state-dict", type=str, default=STATE_DICT_PATH)
    argparser.add_argument("--log-file", type=str, default="", help="Log file name")
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--inference", action="store_true")
    argparser.add_argument("--perf", action="store_true")
    argparser.add_argument("--norm", type=str, default="right")
    argparser.add_argument('--num-workers', type=int, default=12,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
    args = argparser.parse_args()
    print(args)

    try:
        from dgl.backend.pytorch.sparse import S
        print(f"Running in customized kernel with S={S}")
    except ImportError as e:
        S = None
        print("Running in default DGL")

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    g, n_classes = load_ogb('ogbn-products')

    in_feats = g.ndata['features'].shape[1]

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    data = in_feats, n_classes, train_g, val_g, test_g

    run(args, device, data)
