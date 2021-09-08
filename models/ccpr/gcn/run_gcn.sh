# training commend, modify parameters for your need
# python train.py --gpu=0 --dataset=pubmed --self-loop --train --log --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=20 --kernel=cuSPARSE --S=0 --best-val 

python train.py --gpu=1 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=10 --n-runs=2 --kernel=cuSPARSE --train --best-val --log 

# python train.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --train --best-val --log 
# python train.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V2 --sr=0.3 --train --best-val --log 

# profile inference
# python train.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=cuSPARSE --prof-infer --best-val --log 
return
