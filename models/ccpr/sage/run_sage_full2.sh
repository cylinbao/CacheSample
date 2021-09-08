# training commend, modify parameters for your need
python train_full2.py --gpu=0 --dataset=pubmed --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --train --log --best-val 

# profile inference
# python train_full2.py --gpu=0 --dataset=pubmed --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --prof-infer --log 
return
