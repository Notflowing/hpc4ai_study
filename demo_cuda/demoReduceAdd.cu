#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include "helper_cuda.h"
#include "ticktockCPUGPU.h"
#include <omp.h>

template <typename T>
void initArray(T* arr, const int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 13;
    }
}

// naive reduce kernel
template <typename T>
__global__ void kernelReduceAdd_naive(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    long long ix = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    if (ix >= n) return;
    // 减少分支分化
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            d_in[ix] += d_in[ix + stride];
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = d_in[ix];
}

template <typename T>
__global__ void kernelBlockReduce_naive(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    int tid = threadIdx.x;
    int iter = n / blockDim.x;
    #pragma unroll
    for (int i = 1; i <= iter; ++i) {
        if (tid + i * blockDim.x < n) {
            d_in[tid] += d_in[tid + i * blockDim.x];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            d_in[tid] += d_in[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) d_out[0] = d_in[tid];
}

template <typename T, int size_block = 1024, int iter = 1>
T reduceAdd_naive(T* d_a, const long long n) {
    T h_sum = 0;
    int num_block = (n + size_block - 1) / size_block;
    T *d_block, *d_sum;
    checkCudaErrors(cudaMalloc((void **)&d_block, sizeof(T) * num_block));
    checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(T)));
    dim3 block(size_block, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, 1, 1);

    for (int i = 0; i < iter; ++i) {
        kernelReduceAdd_naive<<<grid, block>>>(d_a, d_block, n);
        kernelBlockReduce_naive<<<dim3(1, 1, 1), dim3(size_block, 1, 1)>>>(d_block, d_sum, num_block);
    }
    checkCudaErrors(cudaMemcpy(&h_sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_block));
    checkCudaErrors(cudaFree(d_sum));
    return h_sum;
}

// smem reduce kernel
template <typename T>
__global__ void kernelReduceAdd_smem(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    long long ix = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    // 使用共享内存，赋初值
    extern __shared__ T sdata[];
    sdata[tid] = 0;

    if (ix >= n) return;
    sdata[tid] = d_in[ix];
    __syncthreads();
    // 减少共享内存的bank冲突，stride从大到小遍历
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride]; 
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__ void kernelBlockReduce_smem(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    int tid = threadIdx.x;
    extern __shared__ T sdata[];
    sdata[tid] = 0;

    if (tid >= n) return;
    sdata[tid] = d_in[tid];
    __syncthreads();

    int iter = n / blockDim.x;
    #pragma unroll
    for (int i = 1; i <= iter; ++i) {
        if (tid + i * blockDim.x < n) {
            sdata[tid] += d_in[tid + i * blockDim.x];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) d_out[0] = sdata[tid];
}


template <typename T, int size_block = 1024, int iter = 1>
T reduceAdd_smem(T* d_a, const long long n) {
    T h_sum = 0;
    int num_block = (n + size_block - 1) / size_block;
    T *d_block, *d_sum;
    checkCudaErrors(cudaMalloc((void **)&d_block, sizeof(T) * num_block));
    checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(T)));
    dim3 block(size_block, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, 1, 1);

    for (int i = 0; i < iter; ++i) {
        kernelReduceAdd_smem<<<grid, block, sizeof(T) * size_block>>>(d_a, d_block, n);
        kernelBlockReduce_smem<<<dim3(1, 1, 1), dim3(size_block, 1, 1), sizeof(T) * size_block>>>(d_block, d_sum, num_block);
    }
    checkCudaErrors(cudaMemcpy(&h_sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_block));
    checkCudaErrors(cudaFree(d_sum));
    return h_sum;
}

// warpshuffle smem reduce kernel
template <typename T>
__device__ __forceinline__ T warpReduceSum(T sum) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

template <typename T>
__global__ void kernelReduceAdd_warp(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    long long ix = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ T sdata[];
    sdata[tid] = 0;

    if (ix >= n) return;
    sdata[tid] = d_in[ix];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride]; 
        }
        __syncthreads();
    }

    T res = 0;
    if (tid < 32) {
        res = sdata[tid] + sdata[tid + 32];
        // 使用warp shuffle指令，在一个线程束内加速计算
        res = warpReduceSum(res);
    }
    if (tid == 0) d_out[blockIdx.x] = res;
}

template <typename T>
__global__ void kernelBlockReduce_warp(T* __restrict__ d_in, T* __restrict__ d_out, const long long n) {
    int tid = threadIdx.x;
    extern __shared__ T sdata[];
    sdata[tid] = 0;

    if (tid >= n) return;
    sdata[tid] = d_in[tid];
    __syncthreads();

    int iter = n / blockDim.x;
    #pragma unroll
    for (int i = 1; i <= iter; ++i) {
        if (tid + i * blockDim.x < n) {
            sdata[tid] += d_in[tid + i * blockDim.x];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    T res = 0;
    if (tid < 32) {
        res = sdata[tid] + sdata[tid + 32];
        res = warpReduceSum(res);
    }
    if (tid == 0) d_out[0] = res;
}

template <typename T, int size_block = 1024, int iter = 1>
T reduceAdd_warp(T* d_a, const long long n) {
    T h_sum = 0;
    int num_block = (n + size_block - 1) / size_block;
    T *d_block, *d_sum;
    checkCudaErrors(cudaMalloc((void **)&d_block, sizeof(T) * num_block));
    checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(T)));
    dim3 block(size_block, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, 1, 1);

    for (int i = 0; i < iter; ++i) {
        kernelReduceAdd_warp<<<grid, block, sizeof(T) * size_block>>>(d_a, d_block, n);
        kernelBlockReduce_warp<<<dim3(1, 1, 1), dim3(size_block, 1, 1), sizeof(T) * size_block>>>(d_block, d_sum, num_block);
    }
    checkCudaErrors(cudaMemcpy(&h_sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_block));
    checkCudaErrors(cudaFree(d_sum));
    return h_sum;
}

int main(int argc, char **argv) {
    int m, n;
    if (argc > 1) {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
    }
    else {
        m = 1024;
        n = 1024;
    }

    long long size = m * n;
    int *h_a = new int[size];
    initArray(h_a, size);

    int *d_a1, *d_a2, *d_a3;
    checkCudaErrors(cudaMalloc((void **)&d_a1, sizeof(int) * size));
    checkCudaErrors(cudaMalloc((void **)&d_a2, sizeof(int) * size));
    checkCudaErrors(cudaMalloc((void **)&d_a3, sizeof(int) * size));
    checkCudaErrors(cudaMemcpy(d_a1, h_a, sizeof(int) * size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_a2, h_a, sizeof(int) * size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_a3, h_a, sizeof(int) * size, cudaMemcpyHostToDevice));

    int sum_CPU = 0, sum_GPU1 = 0, sum_GPU2 = 0, sum_GPU3 = 0;

    // naive reduceAdd
    std::cout << "====naive reduceAdd====" << std::endl;
    Ticktock redAdd_naive("reduceAdd_naive");
    const int iter = 1;

    redAdd_naive.tickCPU();
    for (int it = 0; it < iter; ++it) {
        #pragma omp parallel for
        for (long long i = 0; i < size; ++i) {
            sum_CPU += h_a[i];
        }
    }
    redAdd_naive.tockCPU();    

    redAdd_naive.tickGPU();
    sum_GPU1 = reduceAdd_naive(d_a1, size);
    redAdd_naive.tockGPU();

    bool flag1 = (sum_CPU == sum_GPU1);
    std::cout << sum_CPU << " ?= " << sum_GPU1 << " : ";
    if (flag1) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END naive reduceAdd

    // SMEM reduceAdd
    std::cout << "====SMEM reduceAdd====" << std::endl;
    Ticktock redAdd_smem("reduceAdd_smem");
    redAdd_smem.tickGPU();
    sum_GPU2 = reduceAdd_smem(d_a2, size);
    redAdd_smem.tockGPU();

    bool flag2 = (sum_CPU == sum_GPU2);
    std::cout << sum_CPU << " ?= " << sum_GPU2 << " : ";
    if (flag2) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END SMEM reduceAdd

    // warpshuffle SMEM reduceAdd
    std::cout << "====Warp reduceAdd====" << std::endl;
    Ticktock redAdd_warp("reduceAdd_warp");
    redAdd_warp.tickGPU();
    sum_GPU3 = reduceAdd_warp(d_a3, size);
    redAdd_warp.tockGPU();

    bool flag3 = (sum_CPU == sum_GPU3);
    std::cout << sum_CPU << " ?= " << sum_GPU3 << " : ";
    if (flag3) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END warpshuffle SMEM reduceAdd

    delete[] h_a;
    checkCudaErrors(cudaFree(d_a1));
    checkCudaErrors(cudaFree(d_a2));
    checkCudaErrors(cudaFree(d_a3));

    return 0;
}