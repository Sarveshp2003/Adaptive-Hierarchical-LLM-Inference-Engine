#include <cuda_runtime.h>
#include <iostream>
#include "CUDAError.h"


__global__
void addKernel(
    const float* A,
    const float* B,
    float* C,
    int elements
)
{

    int index =
        blockIdx.x * blockDim.x + threadIdx.x;


    if(index < elements)
    {
        C[index] =
            A[index] +
            B[index];
    }

}



extern "C"
void launchAdd(

    float* A,

    float* B,

    float* C,

    int elements

)
{

    int threads = 256;


    int blocks =
        (elements + threads - 1)
        / threads;


    addKernel<<<
        blocks,
        threads,
        0,
        0
    >>>(
        A,
        B,
        C,
        elements
    );


    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

}