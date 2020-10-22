# able to achieve 96.077% accuracy with mean aggregator on reddit dataset

# training command
python train_full.py --gpu=0 --dataset=reddit --n-hidden=256 --aggregator-type=gcn --train

# inference command for cusparse
python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=gcn --inference

# inference command for cache sample
# python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=gcn --inference --cache-sample
