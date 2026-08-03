#include <iostream>

#include "Tensor.h"
#include "TensorOps.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"


int main()
{

    RuntimeMemory::initializeGPU(
        1024 * 1024 * 1024
    );


    CUDAStream::initialize();


    Tensor A(
        {4},
        DataType::FP32
    );


    Tensor B(
        {4},
        DataType::FP32
    );


    Tensor C(
        {4},
        DataType::FP32
    );


    A.allocateCPU();
    B.allocateCPU();


    for(int i=0;i<4;i++)
    {
        A.cpu()[i]=1.0f;
        B.cpu()[i]=2.0f;
    }


    A.allocateGPU();
    B.allocateGPU();
    C.allocateGPU();


    A.upload();
    B.upload();


    add(
        A,
        B,
        C
    );


    CUDAStream::synchronize();


    C.download();


    std::cout<<"Result:\n";


    for(int i=0;i<4;i++)
    {
        std::cout
        << C.cpu()[i]
        << " ";
    }


    std::cout<<std::endl;


    RuntimeMemory::shutdown();


    std::cout
        <<"ADD TEST PASSED"
        <<std::endl;


    return 0;
}