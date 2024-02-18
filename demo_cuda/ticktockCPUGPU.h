#ifndef TICKTOCK_CPUGPU_H_
#define TICKTOCK_CPUGPU_H_

#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include "helper_cuda.h"

class Ticktock {
private:
    float benchCPU = 0.0f;
    float benchGPU = 0.0f;
    std::string strDemo{};
    std::chrono::steady_clock::time_point startCPU;
    std::chrono::steady_clock::time_point stopCPU;
    cudaEvent_t startGPU;
    cudaEvent_t stopGPU;

public:
    Ticktock() = default;
    Ticktock(const std::string& strDemo): strDemo(strDemo) {
        checkCudaErrors(cudaEventCreate(&startGPU));
        checkCudaErrors(cudaEventCreate(&stopGPU));
    }
    void tickCPU() {
        startCPU = std::chrono::steady_clock::now();
    }
    void tockCPU() {
        stopCPU = std::chrono::steady_clock::now();
        benchCPU = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1000>>>(stopCPU - startCPU).count();
        std::cout << strDemo + "CPU: " << benchCPU << "ms" << std::endl;
    }

    void tickGPU() {
        checkCudaErrors(cudaEventRecord(startGPU));
    }
    void tockGPU() {
        checkCudaErrors(cudaEventRecord(stopGPU));
        checkCudaErrors(cudaEventSynchronize(stopGPU));
        checkCudaErrors(cudaEventElapsedTime(&benchGPU, startGPU, stopGPU));
        std::cout << strDemo + "GPU: " << benchGPU << "ms" << std::endl;
    }

    ~Ticktock() {
        checkCudaErrors(cudaEventDestroy(startGPU));
        checkCudaErrors(cudaEventDestroy(stopGPU));
    }
};

#endif  // TICKTOCK_CPUGPU_H_