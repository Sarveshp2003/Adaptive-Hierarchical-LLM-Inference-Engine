#include "Attention.h"
#include "Tensor.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>


int main()

{

    std::cout
        << "Attention Test"
        << std::endl;


    RuntimeMemory::initializeGPU(
        1024ULL * 1024ULL * 1024ULL
    );


    CUDAStream::initialize();



    Tensor Q(
        {4,4},
        DataType::FP32
    );


    Tensor K(
        {4,4},
        DataType::FP32
    );


    Tensor V(
        {4,4},
        DataType::FP32
    );


    Tensor output(
        {4,4},
        DataType::FP32
    );



    Q.allocateGPU();
    K.allocateGPU();
    V.allocateGPU();
    output.allocateGPU();



    Attention attention(4);



    attention.forward(

        Q,
        K,
        V,
        output

    );


    std::cout
        << "ATTENTION TEST PASSED"
        << std::endl;



    CUDAStream::shutdown();

    RuntimeMemory::shutdown();


    return 0;

}