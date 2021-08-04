# able to achieve 94.45% accuracy on reddit dataset

# training commend, modify parameters for your need
# python train.py --gpu=0 --dataset=reddit --n-hidden=256 --n-layers=1 --self-loop --n-epochs=5 --n-runs=1 --train --kernel=cuSPARSE # --save-model

python train.py --gpu=3 --dataset=reddit --n-hidden=256 --n-layers=1 --self-loop --train --n-epochs=200 --n-runs=20 --kernel=CacheSampleV4 --S=1024 --save-model --log
return

# inference commend, for the original kernel
# python train.py --gpu=0 --dataset=reddit --n-hidden=256 --n-layers=1 --self-loop --inference --kernel=cuSPARSE

# python train.py --gpu=0 --dataset=reddit --n-hidden=256 --n-layers=1 --self-loop --inference --kernel=CacheSampleV4 --S=512 --norm-bias=0


run_cmd () {
    python train.py --gpu=0 --dataset=reddit --n-hidden=128 --n-layers=1 --self-loop --inference --cache-sample --log=$1
}

slist=(16 32 64 128 256 512 1024)

sed -i "s/\/\/ \#define USE_CACHE_SAMPLE/\#define USE_CACHE_SAMPLE/" spmm.cu
# for s in "${slist[@]}"
# do
#     # create temporary sparse.py file with different s for DGL
#     cp -f sparse.py sparse_backup.py
#     sed -i "s/S = 128/S = ${s}/" sparse.py
#     # install custom DGL to sample-cache virtualenv
#     zsh setup.sh
#     run_cmd simrand
#     cp -f sparse_backup.py sparse.py
#     rm sparse_backup.py
# done

# sed -i "s/\#define USE_CACHE_SAMPLE/\/\/ \#define USE_CACHE_SAMPLE/" spmm.cu
# zsh setup.sh

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
