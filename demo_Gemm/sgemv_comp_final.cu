#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <algorithm>

// #include <chrono>
// #define TICK(x) auto bench_##x = std::chrono::steady_clock::now()
// #define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::duration<double>>   \
//                 (std::chrono::steady_clock::now() - bench_##x).count() << "s" << std::endl

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define checkCudaErrors(call)                                                  \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define BLOCK_X 32
#define BLOCK_Y 4

__global__ void naive_gemv_withoutSmem(float * __restrict__ A, float * __restrict__ x, float * __restrict__ y,
                                       const int M, const int N) {
    int ty = blockDim.x * blockIdx.x + threadIdx.x;
    if (ty < M) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            sum += A[ty * N + i] * x[i];
        }
        y[ty] = sum;
    }
}


__global__ void naive_gemv_withSmem(float * __restrict__ A, float * __restrict__ x, float * __restrict__ y,
                                    const int M, const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;
    __shared__ float sdata[BLOCK_Y][BLOCK_X];
    // int smem_row = current_row % BLOCK_Y;

    if(current_row < M){
        float res = 0;
        int kIteration = N / warp_size;
        if (kIteration == 0) kIteration = 1;
        #pragma unroll
        for(int i = 0; i < kIteration; i++){
            int current_col = i * warp_size + laneId;
            res += A[current_row*N + current_col] * x[current_col];
        }

        sdata[ty][laneId] = res;
        __syncthreads();
        #pragma unroll
        for (int s = 1; s < warp_size; s *= 2) {
            // if (laneId % (2 * s) == 0) {
            //     sdata[ty][laneId] += sdata[ty][laneId + s];
            // }
            int index = 2 * s * laneId;
            if (index < warp_size) {
                sdata[ty][index] += sdata[ty][index + s];
            }    
            __syncthreads();
        }

        if(laneId==0) y[current_row] = sdata[ty][laneId];

    }
}


template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
#pragma unroll
    for (int offset = (WarpSize >> 1); offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

// if N == 32
__global__ void Sgemv_v0( 
    float * __restrict__ A,
    float * __restrict__ x,
    float * __restrict__ y, 
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M){
        float res = 0;
        int kIteration = N / warp_size;
        if (kIteration == 0) kIteration = 1;
        #pragma unroll
        for(int i = 0; i < kIteration; i++){
            int current_col = i * warp_size + laneId;
            res += A[current_row*N + current_col] * x[current_col];
        }
        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row] = res;
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);
    float* h_y2 = (float*)malloc(bytes_y);
    float* h_y3 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    // 生成A的数据
    for( int i = 0; i < M * N; i++ ) {
        h_A[i] = (float)i/N;
    }

    // 生成x的数据
    for( int i = 0; i < N; i++ ) {
        h_x[i] = 1;
    }
    memset(h_y, 0, M*sizeof(float));
    memset(h_y1, 0, M*sizeof(float));

    // int nIter = 1000;
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    // TICK(mygemv);

    double msecPerMatrixMul[4] = {0, 0, 0, 0};
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    // Opitim Gemv
    checkCudaErrors(cudaEventRecord(start));
    dim3 dimBlock(32,4);
    dim3 dimGrid(M / 4);
    for (int run = 0 ; run < nIter; run ++ ) {
        Sgemv_v0<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }
    // TOCK(mygemv);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[0] = msecTotal / nIter;

    printf( "mysgemv Time= %.6f msec\n", msecPerMatrixMul[0]);

    // naive Gemv without SMEM
    checkCudaErrors(cudaEventRecord(start));
    dim3 dimBlock_naive(128);
    dim3 dimGrid_naive(M / dimBlock_naive.x);
    for (int run = 0 ; run < nIter; run ++ ) {
        naive_gemv_withoutSmem<<< dimGrid_naive, dimBlock_naive >>>(d_A, d_x, d_y, M, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[1] = msecTotal / nIter;

    printf( "nasgemv Time= %.6f msec\n", msecPerMatrixMul[1]);

    // naive Gemv with SMEM
    checkCudaErrors(cudaEventRecord(start));
    dim3 dimBlock_smem(32, 4);
    dim3 dimGrid_smem(M / 4);
    for (int run = 0 ; run < nIter; run ++ ) {
        naive_gemv_withSmem<<< dimGrid_smem, dimBlock_smem >>>(d_A, d_x, d_y, M, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y2, d_y, bytes_y, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[2] = msecTotal / nIter;

    printf( "naismem Time= %.6f msec\n", msecPerMatrixMul[2]);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    // TICK(cublas);
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_A, N, d_x, 1, &beta, d_y, 1
        );
    }
    // TOCK(cublas);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y3, d_y, bytes_y, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[3] = msecTotal / nIter;
    printf( "cublas  Time= %.6f msec\n", msecPerMatrixMul[3]);
    cublasDestroy(blas_handle);
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = std::max( {fabs(h_y[i] - h_y3[i]), fabs(h_y1[i] - h_y3[i]), fabs(h_y2[i] - h_y3[i])} );

        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y2[i], h_y3[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("mysgemv ratio= %f\n", msecPerMatrixMul[3] / msecPerMatrixMul[0]);
    printf("naivemv ratio= %f\n", msecPerMatrixMul[3] / msecPerMatrixMul[1]);
    printf("naismem ratio= %f\n", msecPerMatrixMul[3] / msecPerMatrixMul[2]);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}