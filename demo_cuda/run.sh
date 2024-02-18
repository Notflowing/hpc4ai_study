#!/bin/bash

set -e

# /public/software/cuda-11.5/bin/nvcc -O3 -std=c++11 -arch=sm_80 -rdc=true  demoMatrixAdd.cu -lgomp -o demoMatrixAdd
# /public/software/cuda-11.5/bin/nvcc -O3 -std=c++11 -arch=sm_80 -rdc=true  demoReduceAdd.cu -lgomp -o demoReduceAdd
# /public/software/cuda-11.5/bin/nvcc -O3 -std=c++11 -arch=sm_80 -rdc=true  demoHistogram.cu -o demoHistogram
/public/software/cuda-11.5/bin/nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true  demoSort.cu -o demoSort