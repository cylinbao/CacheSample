# able to achieve 69.96% test accuracy on ogb-product dataset

# training command
python train_sampling.py --gpu=0 --dataset=ogb-product --num-hidden=256 --batch-size=64 --inductive --train

# inference command
python train_sampling.py --gpu=0 --dataset=ogb-product --num-hidden=256 --batch-size=32768 --inductive --inference
