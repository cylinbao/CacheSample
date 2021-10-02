
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.1 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.2 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.3 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.4 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.5 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.6 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.7 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.8 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=0.9 --log 
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --drop-edge --sr=1.0 --log 

return

# drop edge training command

python train.py --gpu=1 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3  --train --early-stop --patience=100 --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --drop-edge --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3  --train --early-stop --patience=100 --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --drop-edge --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3  --train --early-stop --patience=100 --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --drop-edge --sr=0.7 --log --save-model
return

# profile inference inference

python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=7 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=15 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=res --n-hidden=64 --n-layers=31 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
return

# profile command

# cora
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log

# citeseer
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log

# pubmed
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
return

# train command

# cora
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model

python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=cora --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# citeseer
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model

python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=citeseer --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# pubmed
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
                                                                                                                                                                                            
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
                                                                                                                                                                                            
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=1 --dataset=pubmed --model-type=res --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

return
