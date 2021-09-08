# training commend, modify parameters for your need
# python train.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=1 --self-loop --train --n-runs=20 --kernel=cuSPARSE --best-val --log

# cuSPARSE hidden 128, layer 2, 4, 8
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.3 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.3 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.3 --train --best-val --log 

python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.5 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.5 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.5 --train --best-val --log 

python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.7 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.7 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=0.7 --train --best-val --log 

python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=1 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=1.0 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=3 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=1.0 --train --best-val --log 
python train_dropedge.py --gpu=0 --dataset=pubmed --self-loop --n-hidden=128 --n-layers=7 --n-epochs=200 --n-runs=25 --kernel=cuSPARSE --sr=1.0 --train --best-val --log 
