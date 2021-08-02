#!/bin/zsh

# Set your DGL root directory here
DGL_DIR="$CACHE_SAMPLE_DIR/dgl"

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
    make -C $DGL_DIR/build -j16
else
    mkdir $DGL_DIR/build
    cd $DGL_DIR/build
    cmake -DUSE_CUDA=ON .. 
    cd $CUD_DIR
    make -C $DGL_DIR/build -j4
fi

if [ $? -eq 1 ]; then
    echo "Error while compiling DGL"
    exit
fi

# install dgl python package
echo "Install DGL python"
cd $DGL_DIR/python
python setup.py install &> /dev/null
echo "Finish Install DGL Python"
cd $CUR_DIR
