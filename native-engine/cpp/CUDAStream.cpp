#include "CUDAStream.h"

#include <iostream>


cudaStream_t CUDAStream::stream = nullptr;


void CUDAStream::initialize()
{

    cudaStreamCreate(&stream);

    std::cout
        << "CUDA Stream initialized\n";

}



cudaStream_t CUDAStream::get()
{

    return stream;

}



void CUDAStream::synchronize()
{

    cudaStreamSynchronize(stream);

}



void CUDAStream::shutdown()
{

    if(stream)
    {

        cudaStreamDestroy(stream);

        stream = nullptr;

    }

}