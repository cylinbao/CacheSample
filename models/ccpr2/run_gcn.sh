# training commend, modify parameters for your need

python train.py --gpu=0 --model-type=gcn --dataset=cora --n-hidden=64 --n-layers=1 --n-epochs=20 --n-runs=2 --kernel=cuSPARSE --train --best-val --log 

# python train.py --gpu=0 --model-type=res --dataset=cora --n-hidden=64 --n-layers=1 --n-epochs=20 --n-runs=2 --kernel=cuSPARSE --train --best-val --log 

# python train.py --gpu=0 --model-type=jkn --dataset=cora --n-hidden=64 --n-layers=1 --n-epochs=20 --n-runs=2 --kernel=cuSPARSE --train --best-val --log 

# python train.py --gpu=0 --model-type=sage --aggregator-type=mean --dataset=cora --n-hidden=64 --n-layers=1 --n-epochs=200 --n-runs=1 --kernel=cuSPARSE --train --best-val --log 

# python train.py --gpu=0 --model-type=gcn --dataset=cora --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --train --best-val --log 

# profile inference
# python train.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --prof-infer --log # --best-val 
return
