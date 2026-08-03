#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Basic CUDA error checking macro. Prints file/line and CUDA error string and throws.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA ERROR:\n"                                 \
                      << "File:\n" << __FILE__ << "\n"               \
                      << "Line:\n" << __LINE__ << "\n"               \
                      << "Error:\n" << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err));             \
        }                                                                  \
    } while(0)

#endif