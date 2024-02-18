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
        arr[i] = rand();
    }
}

template <typename T>
bool checkResult(T* a, T* b, const int n) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > 1.0e-6) {
            return false;
        }
        // std::cout << a[i] << "=" << b[i] << std::endl;
    }
    return true;
}

template <typename T>
__global__ void kernelMatrixAdd(T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, 
                                const int m, const int n) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < n && iy < m) {
        long long idx = iy * n + ix;
        c[idx] = a[idx] + b[idx];
    }
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
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c1 = new float[size];
    float *h_c2 = new float[size];

    initArray(h_a, size);
    initArray(h_b, size);

    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void **)&d_a, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void **)&d_c, sizeof(float) * size));

    checkCudaErrors(cudaMemcpy(d_a, h_a, sizeof(float) * size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float) * size, cudaMemcpyHostToDevice));

    Ticktock matAdd("matrixAdd");
    const int iter = 100;
    dim3 block(32, 32, 1);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);
    matAdd.tickGPU();
    for (int i = 0; i < iter; ++i) {
        kernelMatrixAdd<<<grid, block>>>(d_a, d_b, d_c, m, n);
    }
    matAdd.tockGPU();
    checkCudaErrors(cudaMemcpy(h_c1, d_c, sizeof(float) * size, cudaMemcpyDeviceToHost));

    matAdd.tickCPU();
    for (int it = 0; it < iter; ++it) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                long long idx = i * n + j;
                h_c2[idx] = h_a[idx] + h_b[idx];
            }
        }
    }
    matAdd.tockCPU();

    bool flag = checkResult(h_c1, h_c2, size);
    if (flag) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c1;
    delete[] h_c2;
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    return 0;
}