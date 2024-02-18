#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <memory>
#include "helper_cuda.h"
#include "ticktockCPUGPU.h"

template <typename T>
void initArray(T* arr, const long long n) {
    srand(time(0));
    for (long long i = 0; i < n; ++i) {
        arr[i] = rand() % 100;
    }
}

template <typename T>
long long partition(T* arr, long long low, long long high) {
    T pivot = arr[low];
    while (low < high) {
        while (low < high && arr[high] >= pivot) --high;
        arr[low] = arr[high];
        while (low < high && arr[low] <= pivot) ++low;
        arr[high] = arr[low];
    }
    arr[low] = pivot;
    return low;
}

template <typename T>
void quickSort(T* arr, long long low, long long high) {
    if (low < high) {
        long long pivotIdx = partition(arr, low, high);
        quickSort(arr, low, pivotIdx - 1);
        quickSort(arr, pivotIdx + 1, high);
    }
}

template <typename T>
void sortCPU(T* arr, const long long n) {
    long long low = 0;
    long long high = n - 1;
    quickSort(arr, low, high);
}

template <typename T>
std::string isSorted(T const *arr, const long long n) {
    for (long long i = 0; i < n-1; ++i) {
        if (arr[i] > arr[i+1]) {
            return "false";
        }
    }
    return "true";
}

template <typename T>
bool checkResult(T const * arr1, T const * arr2, const long long n) {
    for (long long i = 0; i < n; ++i) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
__global__ void kernelSortOddeven_naive(T* arr, const long long threadNum, 
                                        const long long n, const int isEven) {
    long long ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= threadNum) return;

    long long idx = 2 * ix + isEven;
    if (idx + 1 >= n) return;

    T a0 = arr[idx];
    T a1 = arr[idx + 1];
    if (a0 > a1) {
        arr[idx] = a1;
        arr[idx + 1] = a0;
    }
}

template <typename T>
void sortOddeven_naive(T* d_arr, T* h_arr, const long long size) {
    dim3 block(1024, 1, 1);
    long long threadNum = size / 2;
    dim3 grid((threadNum + block.x - 1) / block.x, 1, 1);
    int isEven = 0;
    // for loop必须在kernel函数外，调度执行odd even排序
    // 如果在kernel函数内部执行for循环，当数据量超过1024时可能会出错
    // 当元素个数小于1024时，使用kernel函数内部的for循环，以及共享内存，是没问题的
    // 但是，元素较少时，CPU端的排序比GPU端更快
    // 考虑归并排序和双调排序
    for (long long count = size; count > 0; --count, ++isEven) {
        isEven %= 2;
        kernelSortOddeven_naive<<<grid, block>>>(d_arr, threadNum, size, isEven);
    }
    checkCudaErrors(cudaMemcpy(h_arr, d_arr, sizeof(T) * size, cudaMemcpyDeviceToHost));
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
    std::unique_ptr<int[]> h_arr = std::make_unique<int[]>(size);       // Pass
    // std::shared_ptr<int[]> h_arr = std::make_shared<int[]>(size);    // Error
    // std::unique_ptr<int[]> h_arr(new int[size]);                     // Pass
    initArray(h_arr.get(), size);

    std::unique_ptr<int[]> d2h_arr1 = std::make_unique<int[]>(size);
    std::unique_ptr<int[]> d2h_arr2 = std::make_unique<int[]>(size);
    std::unique_ptr<int[]> d2h_arr3 = std::make_unique<int[]>(size);
    int *d_arr1, *d_arr2, *d_arr3;
    checkCudaErrors(cudaMalloc((void **)&d_arr1, sizeof(int) * size));
    checkCudaErrors(cudaMalloc((void **)&d_arr2, sizeof(int) * size));
    checkCudaErrors(cudaMalloc((void **)&d_arr3, sizeof(int) * size));
    checkCudaErrors(cudaMemcpy(d_arr1, h_arr.get(), sizeof(int) * size, cudaMemcpyHostToDevice));    
    checkCudaErrors(cudaMemcpy(d_arr2, h_arr.get(), sizeof(int) * size, cudaMemcpyHostToDevice));    
    checkCudaErrors(cudaMemcpy(d_arr3, h_arr.get(), sizeof(int) * size, cudaMemcpyHostToDevice));    

    // naive sort
    std::cout << "====naive sort====" << std::endl;
    Ticktock sort_naive("sort_naive");

    sort_naive.tickCPU();
    sortCPU(h_arr.get(), size);
    sort_naive.tockCPU();
    // std::cout << isSorted(h_arr.get(), size) << std::endl;

    sort_naive.tickGPU();
    sortOddeven_naive(d_arr1, d2h_arr1.get(), size);
    sort_naive.tockGPU();
    // std::cout << isSorted(d2h_arr1.get(), size) << std::endl;

    bool flag1 = checkResult(h_arr.get(), d2h_arr1.get(), size);
    if (flag1) std::cout << "Pass" << std::endl;
    else std::cout << "Error" << std::endl;
    // END naive sort

    // for (long long i = 0; i < size; ++i) {
    //     std::cout << h_arr[i] << " ?= "<< d2h_arr1[i] << (h_arr[i] == d2h_arr1[i] ? " TRUE" : " ERROR") << std::endl;
    // }

    checkCudaErrors(cudaFree(d_arr1));
    checkCudaErrors(cudaFree(d_arr2));
    checkCudaErrors(cudaFree(d_arr3));

    return 0;
}