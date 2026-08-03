#pragma once

#include <cuda_runtime.h>


class CUDAStream
{

public:

    static void initialize();

    static cudaStream_t get();

    static void synchronize();

    static void shutdown();


private:

    static cudaStream_t stream;

};