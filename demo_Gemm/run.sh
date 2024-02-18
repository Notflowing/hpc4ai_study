#!/bin/bash

set -e

/public/software/cuda-11.5/bin/nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true -I/data0/home/wjl/cutlass/include sgemm_comp_final.cu -lcublas -o main

rm log
./main 256 256 256 >> log
./main 512 512 512 >> log
./main 1024 1024 1024 >> log
./main 1536 1536 1536 >> log
./main 2048 2048 2048 >> log
./main 3072 3072 3072 >> log
./main 4096 4096 4096 >> log
./main 6144 6144 6144 >> log
./main 8192 8192 8192 >> log

# ./main 4096 2048 4096 | tee log

# /public/software/cuda-11.5/bin/nvcc -std=c++17 -arch=sm_80 -rdc=true -I/data0/home/wjl/cutlass/include test.cu -o main
# /public/software/cuda-11.5/bin/nvcc -std=c++17 -arch=sm_80 -rdc=true -I/data0/home/wjl/cutlass/include gemm_v1_1.cu -o main

# /public/software/cuda-11.5/bin/nvcc -std=c++17 -arch=sm_80 -rdc=true sgemv_comp_final.cu -lcublas -o main
# ./main 1024 128 | tee log