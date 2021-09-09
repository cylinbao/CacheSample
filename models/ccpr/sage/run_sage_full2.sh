# training commend, modify parameters for your need
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --train --log --best-val 

python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --train --log --best-val 

python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --train --log --best-val 

python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --train --log --best-val 
python train_full2.py --gpu=2 --dataset=citeseer --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --train --log --best-val 

# profile inference
# python train_full2.py --gpu=0 --dataset=pubmed --self-loop --aggregator-type=mean --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=50 --kernel=cuSPARSE --prof-infer --log 
return
