#include <cuda_runtime.h>
#include <iostream>

#include "CUDAStream.h"
#include "Tensor.h"
#include "CUDAError.h"



__global__
void matmulKernel(

    const float* A,

    const float* B,

    float* C,

    int M,

    int K,

    int N

)
{

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;


    int col =
        blockIdx.x * blockDim.x + threadIdx.x;



    if(row < M && col < N)
    {

        float sum = 0.0f;


        for(int k = 0; k < K; k++)
        {

            sum +=
                A[row * K + k] *
                B[k * N + col];

        }


        C[row * N + col] = sum;

    }

}





extern "C"
void launchMatmul(

    float* A,

    float* B,

    float* C,

    int M,

    int K,

    int N

)
{


    dim3 threads(
        16,
        16
    );


    dim3 blocks(

        (N + threads.x - 1) / threads.x,

        (M + threads.y - 1) / threads.y

    );



    matmulKernel<<<

        blocks,

        threads,

        0,

        CUDAStream::get()

    >>>(

        A,

        B,

        C,

        M,

        K,

        N

    );



    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif


}





