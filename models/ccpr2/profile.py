import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from model_utils import load_model

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

    args.n_runs = 50
    times = []
    with profiler.profile(use_cuda=True) as prof:
        for i in range(args.n_runs):
            t = inference(model, g, features, norm_type, args.norm_bias, 
                    args.kernel, args.S)
            times.append(t)
            # print("Inference time: {:.3f}".format(t))
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

def prof_train(args, model, g, features, train_mask, labels, norm_type):
    loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    dur = []
    for e in range(args.n_epochs):
        model.train()

        t0 = time.time()
        logits = model(g, features, norm_type=norm_type, norm_bias=args.norm_bias, 
                    kernel=args.kernel, S=args.S, sample_rate=args.sr)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dur.append(time.time() - t0)

    avg_epoch_t = np.mean(dur[5:]) * 1000
    std_epoch_t = np.std(dur[5:]) * 1000
    print("Avg Epoch Time (ms): {:.3f}".format(avg_epoch_t))

    with profiler.profile(use_cuda=True) as prof:
        for e in range(args.n_epochs):
            model.train()
            logits = model(g, features, norm_type=norm_type, norm_bias=args.norm_bias, 
                        kernel=args.kernel, S=args.S, sample_rate=args.sr)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    events = prof.key_averages()
    avg_spmm_t = 0.0
    avg_mm_t = 0.0
    cuda_total = 0.0
    for evt in events:
        cuda_total += evt.self_cuda_time_total
        if evt.key == "GSpMM":
            avg_spmm_t += evt.self_cuda_time_total/args.n_epochs/1000
        elif evt.key == "aten::mm":
            avg_mm_t += evt.self_cuda_time_total/args.n_epochs/1000
        elif evt.key == "aten::matmul":
            avg_mm_t += evt.self_cuda_time_total/args.n_epochs/1000
    avg_cuda_t = cuda_total/args.n_epochs/1000
    
    print("Avg SpMM CUDA Time (ms): {:.3f}".format(avg_spmm_t))
    print("Avg GEMM CUDA Time (ms): {:.3f}".format(avg_mm_t))

    return avg_epoch_t, std_epoch_t, avg_spmm_t, avg_mm_t
