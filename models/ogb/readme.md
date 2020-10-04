# Running OGB examples
## Preparing Python environments
Create two virtual environments using following command:
```bash
virtualenv -p /usr/bin/python3.7 base
virtualenv -p /usr/bin/python3.7 custom
```
Install the following packages to both environments:
* [pytorch](https://pytorch.org/get-started/locally/)
* [ogb](https://ogb.stanford.edu/docs/home/)

Install DGL to the `base` virtual envornment.
## Setting up enviroments variable
Include CUDA 11.0 path in `PATH`.
Setup `LD_LIBRARY_PATH` with 
```
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0/lib:${LD_LIBRARY_PATH}
```
and set `DGL_DIR` to your DGL repo.
