# training commend, modify parameters for your need
# python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=3 --self-loop --train --n-runs=10 --kernel=cuSPARSE # --save-model

# cuSPARSE hidden 128, layer 2, 4, 8
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 

# CacheSampleV3 hidden 128, layer 2
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val
return

# CacheSampleV3 hidden 128, layer 4
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val

# CacheSampleV3 hidden 128, layer 8
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=2 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val 
return
                                                                                        
# cuSPARSE hidden 256, layer 2, 4, 8
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 

# CacheSampleV3 hidden 256, layer 2
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=3 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val
return

# CacheSampleV3 hidden 256, layer 2
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val 
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=3 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val

# CacheSampleV3 hidden 256, layer 2
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=16 --best-val
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=32 --best-val
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=64 --best-val
python train.py --gpu=1 --dataset=pubmed --self-loop --train --log --n-hidden=256 --n-layers=7 --n-epochs=200 --n-runs=20 --kernel=CacheSampleV3 --S=128 --best-val
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
