# Running benchmark with `virtualenv`
Use `zsh benchmark.sh sage`
# Running benchmark with `make`
Use `make train-sage` to generate parameter or inference.

Use `make inference-sage-samplecache` to run inference.

Note that before running make `inference-sage-samplecache` make sure your kernel has been replaced with ones that normalization disabled. 
You can use `make install-sage` to install DGL with modified kernel to your Python.
`S` can also be adjusted with `make install-sage S=your_s`. 

## Parameters to tweak
Add `NO_VIRTUALENV=1` to use your default Python interpreter

Add `NO_PERF=1` to disable CUDA profiling
### Example
`make inference-sage-samplecache NO_VIRTUALENV=1 NO_PERF=1` to run inferece without virtualenv and disabling CUDA profiling

