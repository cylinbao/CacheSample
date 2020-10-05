# able to achieve ??% accuracy on ogb-product dataset

# training command
python train_sampling.py --gpu=0 --dataset=ogb-product --num-hidden=256 --batch-size=16384 --inductive --train --num-epoch=1 > train_sampl.log

if [ $? -eq 1 ]; then
    exit
fi

# inference command
python train_sampling.py --gpu=0 --dataset=ogb-product --num-hidden=256 --batch-size=32768 --inductive --inference > infer_sampl.log

sudo poweroff
