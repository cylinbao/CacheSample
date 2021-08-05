# training command
# python train_full.py --gpu=0 --dataset=reddit --n-hidden=16 --aggregator-type=mean --train --save-model

# python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --S=0 --early-stop --log

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --S=0 --early-stop --log

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --S=0 --early-stop --log

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --S=0 --early-stop --log
return

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --S=0 --early-stop

# CacheSample, 256, 1
python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=128 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=256 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=512 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=1024 --early-stop

# CacheSample, 512, 1
python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=128 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=256 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=512 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=1024 --early-stop

# CacheSample, 256, 4
python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=128 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=256 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=512 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=256  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=1024 --early-stop

# CacheSample, 512, 4
python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=128 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=256 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=512 --early-stop

python train_full.py --gpu=0 --dataset=reddit --aggregator-type=mean --train --n-hidden=512  --n-layers=3 --n-epochs=200 --n-runs=5 --kernel=CacheSampleV4 --S=1024 --early-stop
return

# inference command for cusparse
python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference --kernel=cuSPARSE # --log=simrand

# inference command for cache sample
# python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference --kernel=CacheSample --S=256 # --log=simrand

python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference --kernel=Sample --S=256 # --log=simrand

run_cmd () {
    python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference --cache-sample --log=$1 
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
