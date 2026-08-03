#include <iostream>
#include <cassert>

#include "MultiHeadAttention.h"
#include "Tensor.h"
#include "RuntimeMemory.h"


int main()
{

    std::cout
        << "MultiHead Attention Test"
        << std::endl;


    RuntimeMemory::initializeGPU(
        1024 * 1024 * 1024
    );


    int hiddenSize = 4;
    int heads = 2;


    MultiHeadAttention mha(
        hiddenSize,
        heads
    );


    std::vector<int> shape =
    {
        1,
        4,
        hiddenSize
    };


    Tensor input(
        shape,
        DataType::FP32
    );


    Tensor output(
        shape,
        DataType::FP32
    );


    input.allocateCPU();
    input.allocateGPU();
    output.allocateCPU();
    output.allocateGPU();


    float values[16] =
    {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };



    //
    // Copy input data
    //
    float* inputPtr =
        input.cpu();



    for(int i=0;i<16;i++)
    {
        inputPtr[i] = values[i];
    }



    input.upload();



    std::cout
        << "Running forward..."
        << std::endl;



    mha.forward(
        input,
        output
    );



    output.download();



    float* outputPtr =
        output.cpu();



    std::cout
        << "Output:"
        << std::endl;



    for(int i=0;i<16;i++)
    {
        std::cout
            << outputPtr[i]
            << " ";
    }


    std::cout
        << std::endl;



    assert(
        output.elements() == 16
    );


    bool changed = false;

    for(int i = 0; i < 16; i++)
    {
        if(outputPtr[i] != values[i])
        {
            changed = true;
            break;
        }
    }


    assert(changed);

    std::cout
        << "Attention computation changed output"
        << std::endl;



    std::cout
        << "MULTIHEAD ATTENTION TEST PASSED"
        << std::endl;



    RuntimeMemory::shutdown();


    return 0;
}