# able to achieve 80.2% accuracy on pubmed dataset

# commends for pubmed
# training command, modify parameters for your need
python train.py --dataset pubmed --gpu 0 --n-layers 16 --n-hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --norm=both --train

# inference command, for the original kernel
python train.py --dataset pubmed --gpu 0 --n-layers 16 --n-hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --norm=both --inference

# inference command, use --norm=none for cache_sample kernel
python train.py --dataset pubmed --gpu 0 --n-layers 16 --n-hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --norm=none --inference

# command for cora, the default parameters are set for cora
# python train.py --dataset cora --gpu 0 --norm=both --train

# command for citeseer, the default parameters are set for cora
# python train.py --dataset citeseer --gpu 0 --n-layers 32 --n-hidden 256 --lamda 0.6 --dropout 0.7 --norm=both --train
