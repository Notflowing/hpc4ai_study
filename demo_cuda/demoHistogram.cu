#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include "helper_cuda.h"
#include "ticktockCPUGPU.h"
#include <omp.h>

const int histogram_size = 26;

// ASCII a-z: 97-122
template <typename T>
void initArray(T* arr, const long long n) {
    srand(time(0));
    for (long long i = 0; i < n; ++i) {
        arr[i] = rand() % 26 + 97;
    }
}

template <typename T>
bool checkResult(T const * a, T const * b) {
    for (int i = 0; i < histogram_size; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
        // std::cout << a[i] << "=" << b[i] << std::endl;
    }
    return true;
}

template <typename T>
__global__ void kernelHistogram_naive(T const * __restrict__ d_str, int * __restrict__ d_histo, const long long size) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= size) return;

    atomicAdd(&d_histo[d_str[ix] - 'a'], 1);
}

template <typename T>
void histogram_naive(T const * d_str, int* d_histo, int* h_histo, const long long size) {
    dim3 block(1024, 1, 1);
    dim3 grid((size + block.x - 1) / block.x, 1, 1);
    kernelHistogram_naive<<<grid, block>>>(d_str, d_histo, size);
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(int) * histogram_size, cudaMemcpyDeviceToHost));
}

// template <typename T, int blockSize = 16>
// __global__ void kernelHistogram_block(T const * __restrict__ d_str, int * __restrict__ d_histo, const long long threadNum, const long long size) {
//     int ix = blockDim.x * blockIdx.x + threadIdx.x;
//     if (ix >= threadNum) return;
//     long long startIdx = ix * blockSize;

//     #pragma unroll
//     for (int i = 0; i < blockSize; ++i) {
//         if (startIdx + i < size) {
//             atomicAdd(&d_histo[d_str[startIdx + i] - 'a'], 1);
//         }
//     }
// }
// 每个线程读取的数据不连续，stride为线程的总数目，这样做的考虑是合并内存访问，增加缓存的命中率
template <typename T, int blockSize = 16>
__global__ void kernelHistogram_block(T const * __restrict__ d_str, int * __restrict__ d_histo, const long long threadNum, const long long size) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= threadNum) return;

    #pragma unroll
    for (int i = 0; i < blockSize; ++i) {
        long long idx = ix + i * threadNum;
        if (idx < size) {
            atomicAdd(&d_histo[d_str[idx] - 'a'], 1);
        }
    }
}

template <typename T>
void histogram_block(T const * d_str, int* d_histo, int* h_histo, const long long size) {
    dim3 block(1024, 1, 1);
    const int block_size = 16;
    long long threadNum = (size + block_size - 1) / block_size;
    dim3 grid((threadNum + block.x - 1) / block.x, 1, 1);
    kernelHistogram_block<T, block_size><<<grid, block>>>(d_str, d_histo, threadNum, size);
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(int) * histogram_size, cudaMemcpyDeviceToHost));
}

// 使用共享内存SMEM，减少原子操作的访存延迟（确实获得了显著的性能提升）
template <typename T, int blockSize = 16>
__global__ void kernelHistogram_smem(T const * __restrict__ d_str, int * __restrict__ d_histo, const long long threadNum, const long long size) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int histo_smem[];
    if (tid < histogram_size) histo_smem[tid] = 0;
    __syncthreads();

    if (ix >= threadNum) return;

    #pragma unroll
    for (int i = 0; i < blockSize; ++i) {
        long long idx = ix + i * threadNum;
        if (idx < size) {
            atomicAdd(&histo_smem[d_str[idx] - 'a'], 1);
        }
        __syncthreads();
    }
    if (tid < histogram_size) atomicAdd(&d_histo[tid], histo_smem[tid]);
}

template <typename T>
void histogram_smem(T const * d_str, int* d_histo, int* h_histo, const long long size) {
    dim3 block(1024, 1, 1);
    const int block_size = 16;
    long long threadNum = (size + block_size - 1) / block_size;
    dim3 grid((threadNum + block.x - 1) / block.x, 1, 1);
    kernelHistogram_smem<T, block_size><<<grid, block, sizeof(int) * histogram_size>>>(d_str, d_histo, threadNum, size);
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(int) * histogram_size, cudaMemcpyDeviceToHost));
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
    char *h_str = new char[size];
    initArray(h_str, size);
    char *d_str;
    checkCudaErrors(cudaMalloc((void **)&d_str, sizeof(char) * size));
    checkCudaErrors(cudaMemcpy(d_str, h_str, sizeof(char) * size, cudaMemcpyHostToDevice));

    int *h_histo0 = new int[histogram_size]();
    int *h_histo1 = new int[histogram_size]();
    int *h_histo2 = new int[histogram_size]();
    int *h_histo3 = new int[histogram_size]();

    int *d_histo1, *d_histo2, *d_histo3;
    checkCudaErrors(cudaMalloc((void **)&d_histo1, sizeof(int) * histogram_size));
    checkCudaErrors(cudaMalloc((void **)&d_histo2, sizeof(int) * histogram_size));
    checkCudaErrors(cudaMalloc((void **)&d_histo3, sizeof(int) * histogram_size));
    checkCudaErrors(cudaMemset(d_histo1, 0, sizeof(int) * histogram_size));
    checkCudaErrors(cudaMemset(d_histo2, 0, sizeof(int) * histogram_size));
    checkCudaErrors(cudaMemset(d_histo3, 0, sizeof(int) * histogram_size));


    // naive histogram
    std::cout << "====naive histogram====" << std::endl;
    Ticktock histo_naive("histogram_naive");
    const int iter = 1;

    histo_naive.tickCPU();
    for (int it = 0; it < iter; ++it) {
        for (long long i = 0; i < size; ++i) {
            h_histo0[h_str[i] - 'a']++;
        }
    }
    histo_naive.tockCPU();    

    histo_naive.tickGPU();
    histogram_naive(d_str, d_histo1, h_histo1, size);
    histo_naive.tockGPU();

    // for (long long i = 0; i < size; ++i) {
    //     std::cout << h_str[i];
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < histogram_size; ++i) {
    //     std::cout << h_histo0[i] << " ?= " << h_histo1[i] << ": " << (h_histo0[i] == h_histo1[i] ? "YES" : "NO") << std::endl;
    // }

    bool flag1 = checkResult(h_histo0, h_histo1);
    if (flag1) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END naive histogram

    // block histogram
    std::cout << "====block histogram====" << std::endl;
    Ticktock histo_block("histogram_block");   

    histo_block.tickGPU();
    histogram_block(d_str, d_histo2, h_histo2, size);
    histo_block.tockGPU();

    bool flag2 = checkResult(h_histo0, h_histo2);
    if (flag2) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END block histogram

    // blockSmem histogram
    std::cout << "====smem histogram====" << std::endl;
    Ticktock histo_smem("histogram_smem");   

    histo_smem.tickGPU();
    histogram_smem(d_str, d_histo3, h_histo3, size);
    histo_smem.tockGPU();

    bool flag3 = checkResult(h_histo0, h_histo3);
    if (flag3) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END blockSmem histogram

    free(h_str);
    free(h_histo0);
    free(h_histo1);
    free(h_histo2);
    free(h_histo3);
    checkCudaErrors(cudaFree(d_str));
    checkCudaErrors(cudaFree(d_histo1));
    checkCudaErrors(cudaFree(d_histo2));
    checkCudaErrors(cudaFree(d_histo3));

    return 0;
}