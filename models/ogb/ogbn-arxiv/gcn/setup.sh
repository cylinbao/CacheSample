#!/bin/zsh

# Set your DGL root directory here
REPO_ROOT="$HOME/gnn_benchmark"
DGL_DIR="$REPO_ROOT/dgl"

# check modification on spmm.cu file
cmp -s ./spmm.cu $DGL_DIR/src/array/cuda/spmm.cu
if [ $? -ne 0 ]; then
    # has changes, so copy spmm.cu and compile 
    echo "Copy ./spmm.cu" to "$DGL_DIR/src/array/cuda/"
    cp ./spmm.cu $DGL_DIR/src/array/cuda/
else
    echo "No changes in spmm.cu"
fi

cmp -s ./csspmm.cuh $DGL_DIR/src/array/cuda/csspmm.cuh
if [ $? -ne 0 ]; then
    echo "Copy ./csspmm.cuh" to "$DGL_DIR/src/array/cuda/"
    cp ./csspmm.cuh $DGL_DIR/src/array/cuda/
else
    echo "No changes in csspmm.cuh"
fi

CUR_DIR=$(pwd)
# check if build dir exist, if not create one and cmake
echo "Compile DGL source"
if [ -d "$DGL_DIR/build" ]; then
    make -C $DGL_DIR/build
else
    mkdir $DGL_DIR/build
    cd $DGL_DIR/build
    cmake -DUSE_CUDA=ON .. 
    cd $CUD_DIR
    make -C $DGL_DIR/build -j8
fi

if [ $? -eq 1 ]; then
    echo "Error while compiling DGL"
    exit
fi

# check modification on sparse.py file
cmp -s ./sparse.py $DGL_DIR/python/dgl/backend/pytorch/sparse.py
if [ $? -eq 1 ]; then
    echo "Copy ./sparse.py to $DGL_DIR/python/dgl/backend/pytorch/"
    cp ./sparse.py $DGL_DIR/python/dgl/backend/pytorch/
else
    echo "No changes in sparse.py"
fi

# install dgl python package
echo "Install DGL python"
cd $DGL_DIR/python
python setup.py install &> /dev/null
echo "Finish Install DGL Python"
cd $CUR_DIR
