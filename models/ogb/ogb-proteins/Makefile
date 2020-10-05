SHELL = /bin/zsh
BASE_PYTHON = base
CUSTOM_PYTHON = sample-cache
N_EPOCH = 500
N_RUNS = 100
AGG = mean
STATE_DICT_NAME = learned_param
HIDDEN = 256
NORM_BASE = right
NORM_CUSTOM = none
S = 128

train-gcn : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=50 \
		--train \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
		--n-hidden=$(HIDDEN)

inference-gcn-base : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
		--n-hidden=$(HIDDEN) \
		--log-file=gcn-base \
		--perf

inference-gcn-samplecache : FORCE
	source ../$(CUSTOM_PYTHON)/bin/activate && \
	python main.py \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_CUSTOM) \
		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
		--n-hidden=$(HIDDEN) \
		--log-file=gcn-$(S) \
		--perf

train-sage : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--sage \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=50 \
		--train \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--n-hidden=$(HIDDEN)

inference-sage-base : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--sage \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--n-hidden=$(HIDDEN) \
		--log-file=sage-base \
		--perf

inference-sage-samplecache : FORCE
	source ../$(CUSTOM_PYTHON)/bin/activate && \
	python main.py \
		--sage \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_CUSTOM) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--n-hidden=$(HIDDEN) \
		--log-file=sage-$(S) \
		--perf

gcn-kernel : FORCE
	cp -f ./spmm-gcn.cu ./spmm.cu

sage-kernel : FORCE
	cp -f ./spmm-sage.cu ./spmm.cu

.PHONY : all $(MAKECMDGOALS)

FORCE :