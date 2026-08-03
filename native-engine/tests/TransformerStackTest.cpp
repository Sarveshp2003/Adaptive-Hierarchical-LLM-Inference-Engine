#include "TransformerStack.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    RuntimeMemory::initializeGPU(1024ULL * 1024ULL * 1024ULL);
    CUDAStream::initialize();

    TransformerStack stack(2, 16);

    Tensor input({1, 1, 16}, DataType::FP32);
    Tensor output({1, 1, 16}, DataType::FP32);

    input.allocateCPU();
    output.allocateCPU();
    input.allocateGPU();
    output.allocateGPU();

    for(int i = 0; i < 16; ++i)
    {
        input.cpu()[i] = static_cast<float>(i + 1);
    }

    input.upload();
    stack.forward(input, output);
    CUDAStream::synchronize();
    output.download();

    bool passed = true;
    for(int i = 0; i < 16; ++i)
    {
        if(!std::isfinite(output.cpu()[i]))
        {
            passed = false;
            break;
        }
    }

    RuntimeMemory::shutdown();
    CUDAStream::shutdown();

    std::cout << (passed ? "TRANSFORMER STACK TEST PASSED" : "TRANSFORMER STACK TEST FAILED") << std::endl;
    return passed ? 0 : 1;
}
