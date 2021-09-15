# training commend, modify parameters for your need
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=cuSPARSE --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=100 --kernel=cuSPARSE --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=15 --n-epochs=200 --n-runs=100 --kernel=cuSPARSE --train --model-type=res --best-val --log 

python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --train --model-type=res --best-val --log 

python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --train --model-type=res --best-val --log 

python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=15 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=15 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --train --model-type=res --best-val --log 
python train2.py --gpu=3 --dataset=cora --self-loop --n-hidden=128 --n-layers=15 --n-epochs=200 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --train --model-type=res --best-val --log 

# profile inference
# python train.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=5 --kernel=cuSPARSE --prof-infer --log # --best-val 
return
