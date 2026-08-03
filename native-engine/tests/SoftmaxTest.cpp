#include <iostream>
#include <cmath>

#include "RuntimeMemory.h"
#include "CUDAStream.h"
#include "Tensor.h"
#include "Softmax.h"

int main()
{
    std::cout
        << "Softmax Test"
        << std::endl;

    RuntimeMemory::initializeGPU(
        1024ULL * 1024ULL * 1024ULL
    );

    CUDAStream::initialize();

    Tensor input(
        {1, 4},
        DataType::FP32
    );

    Tensor output(
        {1, 4},
        DataType::FP32
    );

    input.allocateCPU();
    input.allocateGPU();

    output.allocateCPU();
    output.allocateGPU();

    // Test vector: [1,1,1,1]
    input.cpu()[0] = 1.0f;
    input.cpu()[1] = 1.0f;
    input.cpu()[2] = 1.0f;
    input.cpu()[3] = 1.0f;

    input.upload();

    Softmax softmax;

    std::cout
        << "Running Softmax..."
        << std::endl;

    softmax.forward(
        input,
        output
    );

    CUDAStream::synchronize();

    output.download();

    std::cout
        << "Output:"
        << std::endl;

    for(int i = 0; i < 4; i++)
    {
        std::cout
            << output.cpu()[i]
            << " ";
    }

    std::cout
        << std::endl;

    bool passed = true;

    for(int i = 0; i < 4; i++)
    {
        if(std::fabs(
            output.cpu()[i] - 0.25f
        ) > 0.01f)
        {
            passed = false;
        }
    }

    if(passed)
    {
        std::cout
            << "SOFTMAX TEST PASSED"
            << std::endl;
    }
    else
    {
        std::cout
            << "SOFTMAX TEST FAILED"
            << std::endl;
    }

    RuntimeMemory::shutdown();

    return passed ? 0 : 1;
}