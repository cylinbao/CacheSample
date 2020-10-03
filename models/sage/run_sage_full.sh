# able to achieve 96.077% accuracy on reddit dataset

# training command
# python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean

# inference command
python train_full.py --gpu=0 --dataset=reddit --n-hidden=128 --aggregator-type=mean --inference
