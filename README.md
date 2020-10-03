# gnn_benchmark

## Setup
Clone the project with `--recurse-submodules`
```
git clone --recurse-submodules https://github.com/uwsampl/gnn_benchmark.git
```
If you already clone the project, update the submodules with
```
git submodule update --init --recursive
```
Build DGL
```
mkdir dgl/build
cd dgl/build
cmake -DUSE_CUDA=ON ..
make -j8
```
Or simply work on the benchmarks inside model directory (setup.sh will handle the DGL setup).

## Usage
- To build a new model, simple copy the directory in models.
The existing files serve as example.

- To adjust S length of cache_sample, please modify the S variable in sparse.py.

- To switch between cache_sample kernel and cusparse kernel, uncomment/comment the CACHE_SAMPLE macro in spmm.cu.
