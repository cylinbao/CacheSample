# training commend, modify parameters for your need
python train_sweep.py --gpu=0 --dataset=reddit --self-loop --train --n-runs=20 --kernel=cuSPARSE --save-model --log --n-epochs=200

# python train_sweep.py --gpu=1 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=16 --save-model --log --n-epochs=200

# python train_sweep.py --gpu=1 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=32 --save-model --log --n-epochs=200

# python train_sweep.py --gpu=1 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=64 --save-model --log --n-epochs=200

# python train_sweep.py --gpu=3 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=128 --save-model --log --n-epochs=200

# python train_sweep.py --gpu=3 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=256 --save-model --log --n-epochs=200

# python train_sweep.py --gpu=3 --dataset=reddit --self-loop --train --n-runs=20 --kernel=CacheSample --S=512 --save-model --log --n-epochs=200

# inference commend, for the original kernel
# python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=3 --self-loop --inference --kernel=cuSPARSE # --log=cusparse
