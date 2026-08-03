#include <iostream>
#include <chrono>
#include <vector>

#include "MultiHeadAttention.h"
#include "Tensor.h"
#include "RuntimeMemory.h"

int main()
{
    RuntimeMemory::initializeGPU(1024 * 1024 * 1024);

    int hiddenSize = 64;
    int heads = 8;
    int batch = 1;
    int seqLen = 64; // typical large-size benchmark

    MultiHeadAttention mha(hiddenSize, heads);

    std::vector<int> shape = {batch, seqLen, hiddenSize};
    Tensor input(shape, DataType::FP32);
    Tensor output(shape, DataType::FP32);

    input.allocateCPU(); input.allocateGPU();
    output.allocateCPU(); output.allocateGPU();

    // fill input
    for(size_t i=0;i<input.elements();i++)
        input.cpu()[i] = (float)(i % 13) * 0.01f;
    input.upload();

    // Warmup GPU
    std::cout << "Benchmark Tensor Shape: "
              << input.shape()[0]
              << " "
              << input.shape()[1]
              << " "
              << input.shape()[2]
              << std::endl;
    for(int i=0;i<10;i++)
        mha.forward(input, output);

    // Measure GPU
    int iters = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;i++)
        mha.forward(input, output);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    std::cout << "GPU avg time (ms): " << ms_gpu << std::endl;

    // Warmup CPU
    for(int i=0;i<5;i++)
        mha.forwardCPU(input, output);

    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<10;i++)
        mha.forwardCPU(input, output);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;

    std::cout << "CPU avg time (ms): " << ms_cpu << std::endl;

    RuntimeMemory::shutdown();
    return 0;
}
