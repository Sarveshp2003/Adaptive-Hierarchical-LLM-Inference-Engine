#include <cuda_runtime.h>
#include "CUDAError.h"


__global__
void reluKernel(
    const float* input,
    float* output,
    int elements
)
{

    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;


    if(idx < elements)
    {

        float value =
            input[idx];


        output[idx] =
            value > 0.0f
            ? value
            : 0.0f;

    }

}



extern "C"
void launchRelu(
    float* input,
    float* output,
    int elements
)
{

    int threads = 256;


    int blocks =
        (elements + threads - 1)
        /
        threads;



    reluKernel<<<blocks,threads>>>(

        input,

        output,

        elements

    );

    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

}