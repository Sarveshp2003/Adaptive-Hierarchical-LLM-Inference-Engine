#include <cuda_runtime.h>
#include <stdio.h>
#include "CUDAError.h"


__global__
void matmulTransposeKernel(

    float* A,
    float* B,
    float* C,

    int M,
    int N,
    int K

)
{

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;


    int col =
        blockIdx.x * blockDim.x + threadIdx.x;



    if(row < M && col < N)
    {

        float sum = 0.0f;


        for(int i = 0; i < K; i++)
        {

            /*
                A : M x K

                B : N x K

                We calculate:

                C = A * B^T

                B transpose access:
                B[col][i]
            */


            sum +=

                A[row * K + i] *

                B[col * K + i];

        }


        C[row * N + col] = sum;

    }

}




extern "C"
void launchMatmulTranspose(

    float* A,

    float* B,

    float* C,

    int M,

    int N,

    int K

)
{


    dim3 block(

        16,

        16

    );


    dim3 grid(

        (N + 15) / 16,

        (M + 15) / 16

    );



    matmulTransposeKernel<<<

        grid,

        block

    >>>(

        A,

        B,

        C,

        M,

        N,

        K

    );



    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

}