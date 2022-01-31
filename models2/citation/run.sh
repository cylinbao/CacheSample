# train with cuSPARSE and save model for inference
for model in gcn res jkn sage
do
    for data in pubmed reddit
    do
        for layer in 2 4 8
        do
            python train.py \
                --gpu=0 \
                --dataset=$data \
                --model=$model \
                --n-hidden=128 \
                --n-layers=$layer \
                --train \
                --best-val \
                --n-epochs=400 \
                --n-runs=100 \
                --kernel=cuSPARSE \
                --log \
                --save-model
        done
    done
done

# train with CacheSample2_V1 and just log accuracy
for model in gcn res jkn sage
do
    for data in pubmed reddit
    do
        for layer in 2 4 8
        do
            for sr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
            do
                python train.py \
                    --gpu=0 \
                    --dataset=$data \
                    --model=$model \
                    --n-hidden=128 \
                    --n-layers=$layer \
                    --train \
                    --best-val \
                    --n-epochs=2 \
                    --n-runs=2 \
                    --kernel=CacheSample2_V1 \
                    --sr=$sr \
                    --log
            done
        done
    done
done

return

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
