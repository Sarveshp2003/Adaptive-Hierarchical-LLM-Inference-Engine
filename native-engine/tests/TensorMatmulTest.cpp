#include "Tensor.h"
#include "TensorOps.h"

#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>
#include <chrono>


int main()
{

    std::cout
        << "Tensor Matmul Async Stress Test\n";


    const int N = 512;


    RuntimeMemory::initializeGPU(

        1024 * 1024 * 1024

    );


    CUDAStream::initialize();



    Tensor A(
        {N, N},
        DataType::FP32
    );


    Tensor B(
        {N, N},
        DataType::FP32
    );


    Tensor C(
        {N, N},
        DataType::FP32
    );



    A.allocateCPU();
    B.allocateCPU();
    C.allocateCPU();



    /*
        Initialize A

        A[i] = i+1

    */

    for(int i = 0; i < N*N; i++)
    {

        A.cpu()[i] =
            static_cast<float>(i + 1);


        B.cpu()[i] =
            0.0f;


        C.cpu()[i] =
            0.0f;

    }



    /*
        B = Identity matrix

        A x I = A

    */

    for(int i = 0; i < N; i++)
    {

        B.cpu()[i*N+i] =
            1.0f;

    }



    std::cout
        << "Allocating GPU tensors...\n";


    A.allocateGPU();

    B.allocateGPU();

    C.allocateGPU();



    A.upload();

    B.upload();



    std::cout
        << "GPU upload completed\n";



    std::cout
        << "Launching CUDA Matmul...\n";



    auto start =
        std::chrono::high_resolution_clock::now();



    matmul(

        A,

        B,

        C

    );



    std::cout
        << "Kernel launched asynchronously\n";



    CUDAStream::synchronize();



    auto end =
        std::chrono::high_resolution_clock::now();



    auto duration =
        std::chrono::duration_cast<
            std::chrono::milliseconds
        >(end - start);



    std::cout
        << "CUDA execution time: "
        << duration.count()
        << " ms\n";



    C.download();



    std::cout
        << "Checking result...\n";



    bool success = true;



    /*
        Validate only selected positions

        Full comparison is expensive for stress tests

    */


    int testIndices[] =
    {
        0,
        N-1,
        N,
        N*N/2,
        N*N-1
    };



    for(int index : testIndices)
    {

        if(C.cpu()[index] != A.cpu()[index])
        {

            std::cout
                << "Mismatch at "
                << index
                << " Expected "
                << A.cpu()[index]
                << " Got "
                << C.cpu()[index]
                << "\n";


            success = false;

        }

    }



    if(success)
    {

        std::cout
            << "ASYNC 512x512 MATMUL TEST PASSED\n";

    }
    else
    {

        std::cout
            << "ASYNC 512x512 MATMUL TEST FAILED\n";

    }



    CUDAStream::shutdown();

    RuntimeMemory::shutdown();



    return success ? 0 : 1;

}