#include <torch/torch.h>
// #include <ATen/ATen.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void packed_accessor_kernel(
    torch::PackedTensorAccessor32<float, 2> foo,
    float* trace) {
  int i = threadIdx.x;
  atomicAdd(trace, foo[i][i]);

//   printf("%f\n", *trace);
}


int main(int argc, char **argv) {
    torch::Tensor foo = torch::arange(25).reshape({5, 5});

    torch::Tensor bar = torch::einsum("ii", foo);

    std::cout << "==> matrix is:\n" << foo << std::endl;
    std::cout << "==> trace of it is:\n" << bar << std::endl;

    // torch::Tensor foo_CPU = torch::arange(25).reshape({5, 5});
    torch::Tensor foo_CPU = torch::rand({12, 12});
    // assert foo is 2-dimensional and holds floats
    auto foo_CPU_a = foo_CPU.accessor<float, 2>();
    float trace_CPU = 0;
    for (int i = 0; i < foo_CPU_a.size(0); i++) {
        trace_CPU += foo_CPU_a[i][i];
    }
    std::cout << "CPU accessor: " << trace_CPU << std::endl;

    std::cout << foo_CPU.device() << std::endl;

    // torch::Tensor foo_GPU = torch::arange(25).reshape({5, 5});
    // assert foo is 2-dimensional and holds floats.
    auto foo_GPU = foo_CPU.to("cuda:0");
    auto foo_GPU_a = foo_GPU.packed_accessor32<float,2>();
    float *trace_GPU = 0;
    // cudaMallocManaged((void**)trace_GPU, sizeof(float));
    cudaMalloc((void **)&trace_GPU, sizeof(float));
    cudaMemset(trace_GPU, 0, sizeof(float));

    packed_accessor_kernel<<<1, 12>>>(foo_GPU_a, trace_GPU);
    // cudaDeviceSynchronize();
    float ans = 0;
    cudaMemcpy(&ans, trace_GPU, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU accessor: " << ans << std::endl;

    return 0;
}

