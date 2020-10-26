# able to achieve 96.077% accuracy with mean aggregator on reddit dataset

# training command
python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --train

# inference command for cusparse
python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference

# inference command for cache sample
# python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference --cache-sample
