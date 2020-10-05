#!/bin/zsh

if [ -z ${DGL_DIR+x} ]; then
    # Set your DGL root directory here
    DGL_DIR="/your/dgl/dir"
else
    # Or as an enviroment
    echo "DGL_DIR is set to '$DGL_DIR'"
fi


# check modification on spmm.cu file
cmp -s ./spmm.cu $DGL_DIR/src/array/cuda/spmm.cu
if [ $? -eq 1 ]; then
    # has changes, so copy spmm.cu and compile 
    echo "Copy ./spmm.cu" to "$DGL_DIR/src/array/cuda/"
    cp -f ./spmm.cu $DGL_DIR/src/array/cuda/
    # rm -rf $DGL_DIR/build
    # mkdir $DGL_DIR/build
    # cmake -S $DGL_DIR -B $DGL_DIR/build -DUSE_CUDA=ON
else
    echo "No changes in spmm.cu"
fi

echo "Compile DGL source"
make -j8 -C $DGL_DIR/build
if [ $$ -eq 1 ]; then
    echo "Error while compiling DGL"
    exit
fi

# check modification on sparse.py file
cmp -s ./sparse-tmp.py $DGL_DIR/python/dgl/backend/pytorch/sparse.py
if [ $? -eq 1 ]; then
    echo "Copy ./sparse-tmp.py to $DGL_DIR/python/dgl/backend/pytorch/"
    cp -f ./sparse-tmp.py $DGL_DIR/python/dgl/backend/pytorch/sparse.py
else
    echo "No changes in sparse.py"
fi

# install dgl python package
echo "Install DGL python"
CUR_DIR=$(pwd)
cd $DGL_DIR/python
which python
python setup.py install &> /dev/null
echo "Finish Install DGL Python"
cd $CUR_DIR
