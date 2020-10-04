SHELL = /bin/zsh
BASE_PYTHON = base
CUSTOM_PYTHON = sample-cache
N_EPOCH = 20
N_RUNS = 2
AGG = mean
STATE_DICT_NAME = learned_param
HIDDEN = 256
NORM_BASE = right
NORM_CUSTOM = none
S = 128

# train-gcn : FORCE
# 	source ../$(BASE_PYTHON)/bin/activate && \
# 	python main.py \
# 		--num-epochs=$(N_EPOCH) \
# 		--num-runs=$(N_RUNS) \
# 		--log-every=500 \
# 		--train \
# 		--norm=$(NORM_BASE) \
# 		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
# 		--num-hidden=$(HIDDEN)

# inference-gcn-base : FORCE
# 	source ../$(BASE_PYTHON)/bin/activate && \
# 	python main.py \
# 		--num-epochs=$(N_EPOCH) \
# 		--num-runs=$(N_RUNS) \
# 		--log-every=500 \
# 		--inference \
# 		--norm=$(NORM_BASE) \
# 		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
# 		--num-hidden=$(HIDDEN) \
# 		--log-file=gcn-base \
# 		--perf

# inference-gcn-samplecache : FORCE
# 	source ../$(CUSTOM_PYTHON)/bin/activate && \
# 	python main.py \
# 		--num-epochs=$(N_EPOCH) \
# 		--num-runs=$(N_RUNS) \
# 		--log-every=500 \
# 		--inference \
# 		--norm=$(NORM_CUSTOM) \
# 		--state-dict=$(STATE_DICT_NAME)_gcn.pt \
# 		--num-hidden=$(HIDDEN) \
# 		--log-file=gcn-$(S) \
# 		--perf

train-sage : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--num-epochs=$(N_EPOCH) \
		--num-runs=$(N_RUNS) \
		--log-every=500 \
		--train \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--num-hidden=$(HIDDEN)

inference-sage-base : FORCE
	source ../$(BASE_PYTHON)/bin/activate && \
	python main.py \
		--num-epochs=$(N_EPOCH) \
		--num-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_BASE) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--num-hidden=$(HIDDEN) \
		--log-file=sage-base
		# --perf

inference-sage-samplecache : FORCE
	source ../$(CUSTOM_PYTHON)/bin/activate && \
	python main.py \
		--num-epochs=$(N_EPOCH) \
		--num-runs=$(N_RUNS) \
		--log-every=500 \
		--inference \
		--norm=$(NORM_CUSTOM) \
		--state-dict=$(STATE_DICT_NAME)_sage.pt \
		--num-hidden=$(HIDDEN) \
		--log-file=sage-$(S)
		# --perf

gcn-kernel : FORCE
	cp -f ./spmm-gcn.cu ./spmm.cu

sage-kernel : FORCE
	cp -f ./spmm-sage.cu ./spmm.cu

.PHONY : all $(MAKECMDGOALS)

FORCE :