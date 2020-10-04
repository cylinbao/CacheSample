#!/bin/zsh
run_benchmark() {
    GNN=$1
    if [ ! -e learned_param_${GNN}.pt ]; then
        echo "model parameters not found"
        make train-${GNN}
    fi

    if [${GNN} -eq == gcn]; then
        Ss="16 32 64 128 256"
    else
        Ss="64 128 256 512 1024"
    fi
    
    for s in 16 32 64 128 256
    do
        # create temporary sparse.py file with different s for DGL
        cp -f sparse.py sparse-tmp.py
        sed -i "s/S = 128/S = ${s}/" sparse-tmp.py
        # install custom DGL to sample-cache virtualenv
        source ../sample-cache/bin/activate
        make ${GNN}-kernel
        zsh setup.sh
        deactivate
        rm sparse-tmp.py
        # run the benchmark
        make -s inference-${GNN}-samplecache S=${s}
    done
    make inference-${GNN}-base
}
if [ $# -eq 0 ]; then
    echo "running both benchmark"
    for GNN in gcn sage
    do
        run_benchmark ${GNN}
    done
    tar -cvf archive.tar *.json
else
    echo "running $1 benchmark"
    run_benchmark $1
    tar -cvf archive-${1}.tar ${1}*.json
fi