# able to achieve 94.45% accuracy on reddit dataset

# training commend, modify parameters for your need
# python train.py --gpu=0 --dataset=reddit --n-hidden=128 --n-layers=1 --norm=right

# inference commend, for the original kernel
python train.py --gpu=0 --dataset=reddit --n-hidden=128 --n-layers=1 --norm=right --inference

# inference commend, use --norm=none for cache_sample kernel
# python train.py --gpu=0 --dataset=reddit --n-hidden=128 --n-layers=1 --norm=none --inference
