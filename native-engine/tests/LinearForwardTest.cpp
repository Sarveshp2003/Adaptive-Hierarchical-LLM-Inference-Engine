#include <iostream>

#include "Linear.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"



int main()
{

    std::cout
        << "Linear Forward Test\n";


    RuntimeMemory::initializeGPU(
        1024*1024*1024
    );


    CUDAStream::initialize();



    Linear linear(
        4096,
        4096
    );



    Tensor input(
        {
            4096,
            4096
        },
        DataType::FP32
    );



    Tensor output(
        {
            4096,
            4096
        },
        DataType::FP32
    );



    input.allocateCPU();
    input.allocateGPU();


    output.allocateCPU();
    output.allocateGPU();



    for(size_t i=0;i<input.elements();i++)
    {
        input.cpu()[i]=1.0f;
    }



    input.upload();



    linear.forward(

        input,

        output

    );



    CUDAStream::synchronize();



    output.download();



    std::cout
        << "Output[0]="
        << output.cpu()[0]
        << std::endl;



    CUDAStream::shutdown();

    RuntimeMemory::shutdown();



    std::cout
        << "LINEAR FORWARD PASSED\n";


    return 0;

}