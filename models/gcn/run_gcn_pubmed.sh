# training commend, modify parameters for your need
python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=3 --self-loop --train --n-runs=10 --kernel=cuSPARSE # --save-model
return

# python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=3 --self-loop --train --n-runs=2 --kernel=CacheSample --S=32 --save-model

# inference commend, for the original kernel
# python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=3 --self-loop --inference --kernel=cuSPARSE # --log=cusparse

# python train.py --gpu=0 --dataset=pubmed --n-hidden=64 --n-layers=1 --self-loop --inference --kernel=CacheSampleV3 --S=16 --norm-bias=0

# python train.py --gpu=0 --dataset=pubmed --n-hidden=32 --n-layers=1 --self-loop --inference --log=simrand

# inference commend, use --norm=none for cache_sample kernel
# python train.py --gpu=3 --dataset=reddit --n-hidden=128 --n-layers=1 --self-loop --inference --cache-sample --log=simrand

run_cmd () {
    python train.py --gpu=0 --dataset=pubmed --n-hidden=32 --n-layers=1 --self-loop --inference --cache-sample --log=$1
}

slist=(16 32 64 128 256 512 1024)

sed -i "s/\/\/ \#define USE_CACHE_SAMPLE/\#define USE_CACHE_SAMPLE/" spmm.cu
for s in "${slist[@]}"
do
    # create temporary sparse.py file with different s for DGL
    cp -f sparse.py sparse_backup.py
    sed -i "s/S = 128/S = ${s}/" sparse.py
    # install custom DGL to sample-cache virtualenv
    zsh setup.sh
    run_cmd simrand
    cp -f sparse_backup.py sparse.py
    rm sparse_backup.py
done

sed -i "s/\/\/ \#define USE_BUCKET/\#define USE_BUCKET/" spmm.cu
for s in "${slist[@]}"
do
    # create temporary sparse.py file with different s for DGL
    cp -f sparse.py sparse_backup.py
    sed -i "s/S = 128/S = ${s}/" sparse.py
    # install custom DGL to sample-cache virtualenv
    zsh setup.sh
    run_cmd bucket
    cp -f sparse_backup.py sparse.py
    rm sparse_backup.py
done

# put the comments back
sed -i "s/\#define USE_CACHE_SAMPLE/\/\/ \#define USE_CACHE_SAMPLE/" spmm.cu
sed -i "s/\#define USE_BUCKET/\/\/ \#define USE_BUCKET/" spmm.cu
zsh setup.sh
