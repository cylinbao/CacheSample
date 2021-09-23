# profile command

# cora
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log

# citeseer
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log

# pubmed
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=cuSPARSE --log 

python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.3 --log

python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.5 --log

python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --prof-train --n-epochs=100 --kernel=CacheSample2_V1 --sr=0.7 --log
return

# train command

# cora
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model

python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=cora --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# citeseer
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model

python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=citeseer --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

# pubmed
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=cuSPARSE --log --save-model

python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
                                                                                                                                                                                            
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
                                                                                                                                                                                            
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=3  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=7  --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train.py --gpu=2 --dataset=pubmed --model-type=jkn --n-hidden=64 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=100 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model

return