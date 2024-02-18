#include "NvInfer.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};

int main() {
  cudaSetDevice(0);
  Logger logger;

  // Create an instance of the builder:
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

  // Create a Network Definition
  uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

  // Add the Input layer to the network
  auto input_data = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 4, 4});
  
  // Add the Convolution layer with hidden layer input nodes, strides and weights for filter and bias.
  std::vector<float> filter(2*2, 1.0);
  nvinfer1::Weights filter_w{nvinfer1::DataType::kFLOAT, filter.data(), 4};
  nvinfer1::Weights bias_w{nvinfer1::DataType::kFLOAT, nullptr, 0};
  auto conv2d = network->addConvolution(
                *input_data, 1, nvinfer1::DimsHW{2, 2}, filter_w, bias_w);
  conv2d->setStride(nvinfer1::DimsHW{1, 1});

  // Add a name for the output of the conv2d layer so that the tensor can be bound to a memory buffer at inference time:
  conv2d->getOutput(0)->setName("output");

  // Mark it as the output of the entire network:
  network->markOutput(*conv2d->getOutput(0));

  // Building an Engine(optimize the network)
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  nvinfer1::IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

  // Prepare input_data
  int32_t inputIndex = engine->getBindingIndex("input");
  int32_t outputIndex = engine->getBindingIndex("output");
  std::vector<float> input(4*4, 1.0);
  std::vector<float> output(3*3);
  void *GPU_input_Buffer_ptr;  // a host ptr point to a GPU buffer
  void *GPU_output_Buffer_ptr;  // a host ptr point to a GPU buffer
  void* buffers[2];
  cudaMalloc(&GPU_input_Buffer_ptr, sizeof(float)*4*4); //malloc gpu buffer for input
  cudaMalloc(&GPU_output_Buffer_ptr, sizeof(float)*3*3); //malloc gpu buffer for output

  cudaMemcpy(GPU_input_Buffer_ptr, input.data(), input.size()*sizeof(float), cudaMemcpyHostToDevice); // copy input data from cpu to gpu


  buffers[inputIndex] = static_cast<void*>(GPU_input_Buffer_ptr);
  buffers[outputIndex] = static_cast<void*>(GPU_output_Buffer_ptr);

  // Performing Inference
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  context->executeV2(buffers);

  // copy result data from gpu to cpu
  cudaMemcpy(output.data(), GPU_output_Buffer_ptr, output.size()*sizeof(float), cudaMemcpyDeviceToHost); 

  // display output
  std::cout << "output is : \n";
  for(auto i : output)
    std::cout << i << " ";
  std::cout << std::endl;

  return 0;
}
