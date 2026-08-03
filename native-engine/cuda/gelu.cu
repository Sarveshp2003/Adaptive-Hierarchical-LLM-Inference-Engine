#include <cuda_runtime.h>
#include "CUDAError.h"

__global__
void geluKernel(const float* input, float* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    while(idx < n)
    {
        float x = input[idx];
        float x3 = x * x * x;
        float t = k * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(t));
        idx += stride;
    }
}

extern "C" void launchGelu(const float* input, float* output, size_t n)
{
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if(blocks > 1024) blocks = 1024;
    geluKernel<<<blocks, threads>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
