# profile inference inference

python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
return

python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3 --prof-infer --n-runs=200 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3 --prof-infer --n-runs=200 --kernel=CacheSample2_V3 --log

return

# profile training time breakdown

python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=200 --kernel=CacheSample2_V1 --log

python train.py --gpu=0 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=20 --kernel=cuSPARSE --log 
python train.py --gpu=0 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=20 --kernel=cuSPARSE --drop-edge --log 
python train.py --gpu=0 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3 --prof-train --n-epochs=20 --kernel=CacheSample2_V1 --log

return

# drop edge training command

python train.py --gpu=2 --dataset=reddit --model=jkn --n-hidden=128 --n-layers=3  --train --best-val --n-epochs=200 --n-runs=10 --kernel=cuSPARSE --drop-edge --sr=0.5 --log --save-model

return

# profile inference inference

python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=cora --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=citeseer --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log

python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=cuSPARSE --log
python train.py --gpu=0 --dataset=pubmed --model=jkn --n-hidden=64 --n-layers=3 --prof-infer --n-runs=100 --kernel=CacheSample2_V3 --log
return

# train command

# cora
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# citeseer
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# pubmed
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

return
