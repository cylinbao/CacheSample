# able to achieve 96.077% accuracy with mean aggregator on reddit dataset

# training command
# python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --train --save-model

# inference command for cusparse
python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference --log
return

# inference command for cache sample
python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference --cache-sample

run_sage () {
    python train_full.py --gpu=0 --dataset=pubmed --n-hidden=16 --aggregator-type=mean --inference --cache-sample --log
}

slist=(16 32 64 128 256 512 1024)
    
for s in "${slist[@]}"
do
    # create temporary sparse.py file with different s for DGL
    cp -f sparse.py sparse_backup.py
    sed -i "s/S = 128/S = ${s}/" sparse.py
    # install custom DGL to sample-cache virtualenv
    zsh setup.sh
    run_sage
    cp -f sparse_backup.py sparse.py
    rm sparse_backup.py
done