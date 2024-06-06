#!/bin/bash

/public/software/cuda-11.5/bin/nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true  implicit_GEMM_conv_fp16.cu -o implicit_GEMM_conv_fp16
./implicit_GEMM_conv_fp16