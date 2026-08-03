#include "Tensor.h"
#include "Linear.h"

#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <iostream>


int main()
{

    std::cout << "Linear Layer Test\n";


    const int INPUT = 4;
    const int OUTPUT = 4;


    RuntimeMemory::initializeGPU(
        1024 * 1024 * 1024
    );


    CUDAStream::initialize();



    Tensor input(
        {1, INPUT},
        DataType::FP32
    );


    Tensor output(
        {1, OUTPUT},
        DataType::FP32
    );



    input.allocateCPU();

    output.allocateCPU();



    /*
        Input:

        [1 2 3 4]

    */

    for(int i = 0; i < INPUT; i++)
    {
        input.cpu()[i] =
            i + 1;
    }



    /*
        Allocate GPU
    */

    input.allocateGPU();

    output.allocateGPU();



    input.upload();



    /*
        Linear layer

        output = input * W

    */

    Linear linear(
        INPUT,
        OUTPUT
    );


    linear.forward(
        input,
        output
    );



    CUDAStream::synchronize();



    output.download();



    std::cout
        << "Output:\n";


    for(int i = 0; i < OUTPUT; i++)
    {
        std::cout
            << output.cpu()[i]
            << " ";
    }


    std::cout
        << "\n";



    bool passed = true;



    for(int i = 0; i < OUTPUT; i++)
    {

        if(output.cpu()[i] == 0)
        {
            passed = false;
        }

    }



    if(passed)
    {
        std::cout
            << "LINEAR TEST PASSED\n";
    }
    else
    {
        std::cout
            << "LINEAR TEST FAILED\n";
    }



    CUDAStream::shutdown();

    RuntimeMemory::shutdown();


    return passed ? 0 : 1;

}