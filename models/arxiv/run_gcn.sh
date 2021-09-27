# command for training gcn

python train_new.py --gpu=0 --model=gcn --n-hidden=128 --n-layers=1 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=cuSPARSE --log --save-model
python train_new.py --gpu=0 --model=gcn --n-hidden=128 --n-layers=3 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=cuSPARSE --log --save-model
python train_new.py --gpu=0 --model=gcn --n-hidden=128 --n-layers=7 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=cuSPARSE --log --save-model
python train_new.py --gpu=0 --model=gcn --n-hidden=128 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=cuSPARSE --log --save-model
python train_new.py --gpu=0 --model=gcn --n-hidden=128 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=cuSPARSE --log --save-model
return

python train_new.py --gpu=1 --model=gcn --n-hidden=128 --n-layers=1 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train_new.py --gpu=1 --model=gcn --n-hidden=128 --n-layers=3 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train_new.py --gpu=1 --model=gcn --n-hidden=128 --n-layers=7 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train_new.py --gpu=1 --model=gcn --n-hidden=128 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
python train_new.py --gpu=1 --model=gcn --n-hidden=128 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.3 --log --save-model
return

python train_new.py --gpu=2 --model=gcn --n-hidden=128 --n-layers=1 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train_new.py --gpu=2 --model=gcn --n-hidden=128 --n-layers=3 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train_new.py --gpu=2 --model=gcn --n-hidden=128 --n-layers=7 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train_new.py --gpu=2 --model=gcn --n-hidden=128 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
python train_new.py --gpu=2 --model=gcn --n-hidden=128 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.5 --log --save-model
return

python train_new.py --gpu=3 --model=gcn --n-hidden=128 --n-layers=1 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train_new.py --gpu=3 --model=gcn --n-hidden=128 --n-layers=3 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train_new.py --gpu=3 --model=gcn --n-hidden=128 --n-layers=7 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train_new.py --gpu=3 --model=gcn --n-hidden=128 --n-layers=15 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
python train_new.py --gpu=3 --model=gcn --n-hidden=128 --n-layers=31 --train --early-stop --n-epochs=1500 --n-runs=50 --kernel=CacheSample2_V1 --sr=0.7 --log --save-model
return
