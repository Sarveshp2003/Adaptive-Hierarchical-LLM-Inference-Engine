#include "FeedForward.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    RuntimeMemory::initializeGPU(1024ULL * 1024ULL * 1024ULL);
    CUDAStream::initialize();

    FeedForward ff(8);

    Tensor input({1, 1, 8}, DataType::FP32);
    Tensor output({1, 1, 8}, DataType::FP32);

    input.allocateCPU();
    output.allocateCPU();
    input.allocateGPU();
    output.allocateGPU();

    for(int i = 0; i < 8; ++i)
    {
        input.cpu()[i] = static_cast<float>(i + 1);
    }

    input.upload();
    ff.forward(input, output);
    CUDAStream::synchronize();
    output.download();

    bool passed = true;
    for(int i = 0; i < 8; ++i)
    {
        if(!std::isfinite(output.cpu()[i]))
        {
            passed = false;
            break;
        }
    }

    RuntimeMemory::shutdown();
    CUDAStream::shutdown();

    std::cout << (passed ? "FEEDFORWARD TEST PASSED" : "FEEDFORWARD TEST FAILED") << std::endl;
    return passed ? 0 : 1;
}
