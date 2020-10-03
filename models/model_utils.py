import torch
import os

def save_model(args, model, filename):
    print("Model's state_dict info:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", 
                model.state_dict()[param_tensor].size())

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    print("Saving model's state_dict as", filename)
    torch.save(model.state_dict(), "./{}/{}".format(args.dir, 
        filename))

def load_model(args, model, filename):
    print("Loading model's state_dict", filename)
    model.load_state_dict(torch.load("./{}/{}".format(args.dir, 
        filename)))
    return model
