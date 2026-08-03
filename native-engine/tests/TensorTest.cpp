#include "Tensor.h"

#include <iostream>

#include "RuntimeMemory.h"
int main()
{
        RuntimeMemory::initializeGPU(

            1024 * 1024 * 1024

        );
    std::cout
        << "Tensor CUDA Test\n";


    Tensor tensor(

        {4096,4096},

        DataType::FP32

    );


    std::cout
        << "Tensor size: "
        << tensor.bytes()
        << " bytes\n";


    tensor.allocateCPU();


    std::cout
        << "CPU allocation successful\n";


    tensor.allocateGPU();


    std::cout
        << "GPU allocation successful\n";


    tensor.release();


    std::cout
        << "Tensor released\n";

    RuntimeMemory::shutdown();

    return 0;
}