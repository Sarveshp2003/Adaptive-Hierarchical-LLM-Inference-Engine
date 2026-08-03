#include "TransformerLayer.h"
#include "Tensor.h"
#include "RuntimeMemory.h"
#include <chrono>
#include <iostream>

int main()
{
    RuntimeMemory::initializeGPU(1024 * 1024 * 1024);

    int numLayers = 4;
    int hidden = 64;
    int heads = 8; // TransformerLayer creates attention with 16 currently; keep hidden consistent
    int batch = 1;
    int seq = 64;

    // Create a small stack
    std::vector<TransformerLayer*> layers;
    for(int i=0;i<numLayers;i++) layers.push_back(new TransformerLayer(hidden));

    std::vector<int> shape = {batch, seq, hidden};
    Tensor input(shape, DataType::FP32);
    Tensor output(shape, DataType::FP32);
    input.allocateCPU(); input.allocateGPU(); output.allocateCPU(); output.allocateGPU();

    for(size_t i=0;i<input.elements();i++) input.cpu()[i] = (float)(i%17) * 0.01f;
    input.upload();

    // Warmup
    for(int i=0;i<5;i++)
    {
        Tensor tmp(shape, DataType::FP32);
        tmp.allocateCPU(); tmp.allocateGPU();
        tmp = input; // use copy operator if exists
        for(int l=0;l<numLayers;l++)
        {
            layers[l]->forward(input, output, nullptr, false);
            // swap
            input = output;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int iter=0;iter<10;iter++)
    {
        for(int l=0;l<numLayers;l++)
        {
            layers[l]->forward(input, output, nullptr, false);
            input = output;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
    std::cout << "Transformer stack avg time (ms): " << ms << std::endl;

    RuntimeMemory::shutdown();
    return 0;
}
