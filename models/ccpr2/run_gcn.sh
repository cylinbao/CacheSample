# profile training time breakdown for CacheSample

python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log
return

# profile inference inference

python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

return

# profile training time breakdown

python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=cora --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=citeseer --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=pubmed --model=gcn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=reddit --model=gcn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

return

# training command, modify parameters for your need

# cora
python train.py --gpu=0 --dataset=cora --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=0 --dataset=cora --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

# citeseer
python train.py --gpu=0 --dataset=citeseer --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=0 --dataset=citeseer --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

# pubmed
python train.py --gpu=0 --dataset=pubmed --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=0 --dataset=pubmed --model-type=gcn --n-hidden=64 --n-layers=1  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

return
