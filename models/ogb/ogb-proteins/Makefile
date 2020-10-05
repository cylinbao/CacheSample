SHELL = /bin/zsh
BASE_PYTHON = source ../base/bin/activate &&
CUSTOM_PYTHON = source ../sample-cache/bin/activate &&
N_EPOCH = 500
N_RUNS = 100
AGG = mean
STATE_DICT_NAME = learned_param
HIDDEN = 256
NORM_BASE = right
NORM_CUSTOM = none
S = 128
USE_PERF = --perf
ifdef NO_PERF
	USE_PERF = --perf
endif
ifdef NO_VIRTUALENV
	undefine BASE_PYTHON
	undefine CUSTOM_PYTHON
endif

train-gcn : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--n-epochs=$(N_EPOCH) \
		--n-runs=$(N_RUNS) \
		--log-every=50 \
		--train \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
		--num-hidden=$(HIDDEN)

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
		$(USE_PERF)

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
		$(USE_PERF)

train-sage : FORCE
	$(BASE_PYTHON) \
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
	$(BASE_PYTHON) \
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
		$(USE_PERF)

inference-sage-samplecache : FORCE
	$(CUSTOM_PYTHON) \
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
		$(USE_PERF)

gcn-kernel : FORCE
	cp -f ./spmm-gcn.cu ./spmm.cu

sage-kernel : FORCE
	cp -f ./spmm-sage.cu ./spmm.cu

install-gcn : gcn-kernel
	cp -f sparse.py sparse-tmp.py
	sed -i "s/S = 128/S = $(S)/" sparse-tmp.py
	zsh setup.sh
	rm -f sparse-tmp.py

install-sage : sage-kernel
	cp -f sparse.py sparse-tmp.py
	sed -i "s/S = 128/S = $(S)/" sparse-tmp.py
	zsh setup.sh
	rm -f sparse-tmp.py

.PHONY : all $(MAKECMDGOALS)

FORCE :