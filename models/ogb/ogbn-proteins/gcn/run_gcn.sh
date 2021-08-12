# command for training gcn
# python gcn.py --gpu=1 --train --save-model --n-runs=7 --dir=./state_dicts/gcn_bias/

# command for regular inference
python gcn.py --gpu=0 --dir=./state_dicts/gcn_bias/ --inference # --log=cusparse --acc_analysis  
return

# command for using cache-sample inference
python gcn.py --gpu=0 --dir=./state_dicts/gcn_bias/ --inference --cache-sample # --log=simrand

run_cmd () {
    python gcn.py --gpu=0 --inference --cache-sample --log=$1 --dir=./state_dicts/gcn_bias/
}

# slist=(16 32 64 128 256 512 1024)
slist=(1280 1536 1792 2048 2304)

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
