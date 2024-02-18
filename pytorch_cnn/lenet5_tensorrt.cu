#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#define checkCudaErrors(call)                                               \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}

// Instantiate the ILogger interface
class Logger: public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kVERBOSE) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    // GPU and CUDA info
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    // Console log
    checkCudaErrors(cudaDriverGetVersion(&driverVersion));
    checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    // ======================================================================
    // TensorRT C++ API
    // ======================================================================
    Logger logger;

    // Create an instance of the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // Create a network definition
    uint32_t flag = 1U << static_cast<uint32_t>
        (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    // 1. Import a model using the ONNX Parser
    // 2. Define a model using TensorRT API (define each layer and import the training parameters)

    // Create an ONNX parser and read the model file
    nvonnxparser::IParser* parser= nvonnxparser::createParser(*network, logger);
    const char* onnx_path = "LeNet5.onnx";
    parser->parseFromFile(onnx_path, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); i++) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // Create a build configuration specifying how TensorRT should optimaize the model
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);

    // Create an optimization profile so that we can specify a range of input dimensions.
    // Here we specify the dynamic batch
    int minBatchSize = 1;
    int optBatchSize = 8;
    int maxBatchSize = 32;
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    nvinfer1::ITensor* input_tensor = network->getInput(0);
    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    input_dims.d[0] = minBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    input_dims.d[0] = optBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    // Building an engine (optimize the model)
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    
    // Safely deleted
    std::cout << "Successful convert onnx to engine!" << std::endl;
    delete parser;
    delete network;
    delete builder;

    // Save the engine
    const char* engine_filename = "LeNet5.engine";
    std::ofstream engineWrite(engine_filename, std::ios::binary);
    if (!engineWrite) {
        std::cout << "Cannot open engine file: " << engine_filename << std::endl;
        exit(1);
    }
    engineWrite.write(static_cast<char*>(serializedModel->data()), serializedModel->size());
    engineWrite.close();
    delete serializedModel;

    std::cout << "=====================================" << std::endl;
    std::cout << "=====================================" << std::endl;

    // // 1. Deserialized engine created using TensorRT;    note: can not delete serializedModel
    // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    // nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

    // 2. Read the serialized engine from disk file.
    std::vector<char> trtModelStream;
    size_t size{0};
    std::ifstream engineRead(engine_filename, std::ios::binary);
    if (engineRead.good()) {
        engineRead.seekg(0, engineRead.end);
        size = engineRead.tellg();
        engineRead.seekg(0, engineRead.beg);
        trtModelStream.resize(size);
        engineRead.read(trtModelStream.data(), size);
        engineRead.close();
    }

    // Create an instance of the runtime interface and obtain an engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);

    // Prepare input and output data
    int32_t inputIndex = engine->getBindingIndex("input");
    int32_t outputIndex = engine->getBindingIndex("output");
    // Get the dimensions of the bindings
    nvinfer1::Dims mInputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims mOutputDims = engine->getBindingDimensions(outputIndex);

    std::cout << "Input tensor dimensions: [";
    for (int i = 0; i < mInputDims.nbDims; ++i) {
        if (i == mInputDims.nbDims - 1) std::cout << mInputDims.d[i] << "]" << std::endl;
        else std::cout << mInputDims.d[i] << ", ";
    }
    std::cout << "Output tensor dimensions: [";
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
        if (i == mOutputDims.nbDims - 1) std::cout << mOutputDims.d[i] << "]" << std::endl;
        else std::cout << mOutputDims.d[i] << ", ";
    }

    // Here we set current batch to 2;
    int curBatchSize = 2;
    std::cout << "Here we set current batch to " << curBatchSize << std::endl;
    size_t inputSize = curBatchSize, outputSize = curBatchSize;
    for (int i = 1; i < mInputDims.nbDims; i++) inputSize *= mInputDims.d[i]; 
    for (int i = 1; i < mOutputDims.nbDims; i++) outputSize *= mOutputDims.d[i]; 
    
    std::vector<float> input(inputSize);
    std::vector<float> output(outputSize);
    std::cout << "Input element number: " << inputSize << std::endl;
    std::cout << "Output element number: " << outputSize << std::endl;
    const char* input_filename = "input_data_lenet5.txt";
    std::ifstream input_data(input_filename);
    if (!input_data) {
        std::cout << "Cannot open input file: " << input_filename << std::endl;
        exit(1);
    }
    float value;
    int dataCount = 0;
    while (input_data >> value) {
        input[dataCount] = value;
        dataCount++;
    }
    input_data.close();

    void *input_Buffer_ptr_dev;
    void *output_Buffer_ptr_dev;
    void* buffers[2];   // CUDA memory space

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaMalloc((void**)&input_Buffer_ptr_dev, sizeof(float) * inputSize));
    checkCudaErrors(cudaMalloc((void**)&output_Buffer_ptr_dev, sizeof(float) * outputSize));
    checkCudaErrors(cudaMemcpyAsync(input_Buffer_ptr_dev, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice, stream));
    buffers[inputIndex] = static_cast<void*>(input_Buffer_ptr_dev);
    buffers[outputIndex] = static_cast<void*>(output_Buffer_ptr_dev);

    // Perform inference
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    context->setOptimizationProfileAsync(0, stream);
    mInputDims.d[0] = curBatchSize;
    // setBindingDimensions() and setInputShapeBinding() for all dynamic input tensors or input shape tensors
    // must be called before either executeV2() or enqueueV2().
    context->setBindingDimensions(0, mInputDims);
    context->enqueueV2(buffers, stream, nullptr);

    // Copy result data from gpu to cpu
    checkCudaErrors(cudaMemcpyAsync(output.data(), output_Buffer_ptr_dev, sizeof(float) * output.size(), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(input_Buffer_ptr_dev));
    checkCudaErrors(cudaFree(output_Buffer_ptr_dev));

    // Display output
    int singleOutSize = outputSize / curBatchSize;
    std::cout << "output is:";
    // for (float i: output) {
    for (int i = 0; i < output.size(); i++) {
        if (i % singleOutSize == 0) std::cout << std::endl;
        std::cout << output[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < curBatchSize; i++) {
        int ans = -1;
        int head = i * singleOutSize;
        int tail = (i + 1) * singleOutSize;
        ans = std::max_element(output.begin() + head, output.begin() + tail) - (output.begin() + head);
        std::cout << "The inference answer of LeNet5 is: " << ans << std::endl;
    }

    return 0;
}