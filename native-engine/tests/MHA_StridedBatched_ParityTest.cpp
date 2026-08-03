#include "MultiHeadAttention.h"
#include "Tensor.h"
#include "CUDAStream.h"
#include "CUDAError.h"
#include "RuntimeMemory.h"
#include <iostream>
#include <cmath>

int main()
{
    // Small deterministic test
    int batch = 1;
    int seq = 4;
    int hidden = 8;
    int heads = 2;

    // Initialize GPU memory pool for tests
    RuntimeMemory::initializeGPU(1024 * 1024 * 1024); // 1GB

    MultiHeadAttention mha(hidden, heads);

    Tensor input({batch, seq, hidden}, DataType::FP32);
    input.allocateCPU();
    input.allocateGPU();

    // fill deterministic pattern
    for(int b=0;b<batch;b++)
    {
        for(int i=0;i<seq;i++)
        {
            for(int h=0;h<hidden;h++)
            {
                input.cpu()[b*seq*hidden + i*hidden + h] = (float)((b+1) * (i+1) * (h+1)) * 1e-3f;
            }
        }
    }
    input.upload();

    Tensor outGPU({batch, seq, hidden}, DataType::FP32);
    outGPU.allocateCPU();
    outGPU.allocateGPU();

    Tensor outGPU2({batch, seq, hidden}, DataType::FP32);
    outGPU2.allocateCPU();
    outGPU2.allocateGPU();

    // Run GPU path
    try {
        mha.forward(input, outGPU, nullptr, false);
    } catch(const std::exception &e) {
        std::cerr << "mha.forward threw: " << e.what() << std::endl;
        return 2;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    outGPU.download();

    // Run CPU fallback path
    try {
        mha.forwardCPU(input, outGPU2, nullptr, false);
    } catch(const std::exception &e) {
        std::cerr << "mha.forwardCPU threw: " << e.what() << std::endl;
        return 3;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    outGPU2.download();

    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    size_t elems = (size_t)batch * seq * hidden;
    for(size_t i=0;i<elems;i++)
    {
        float a = outGPU.cpu()[i];
        float b = outGPU2.cpu()[i];
        float absErr = fabsf(a - b);
        float rel = fabsf(absErr / (b == 0.0f ? 1.0f : b));
        if(absErr > maxAbs) maxAbs = absErr;
        if(rel > maxRel) maxRel = rel;
    }

    std::cout << "Max abs error: " << maxAbs << "  Max rel error: " << maxRel << std::endl;

    const float absTol = 1e-3f;
    const float relTol = 1e-3f;

    if(maxAbs > absTol && maxRel > relTol)
    {
        std::cerr << "TEST FAILED: outputs differ beyond tolerances\n";
        return 1;
    }

    std::cout << "TEST PASSED\n";

    RuntimeMemory::shutdown();
    return 0;
}
