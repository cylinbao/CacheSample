# training command
# python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --train --save-model

# inference command for cusparse
python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference # --log=simrand
return

# inference command for cache sample
# python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference --log=simrand --cache-sample

run_cmd () {
    python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference --log=$1 --cache-sample 
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
