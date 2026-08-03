#include "LayerNorm.h"
#include "Tensor.h"
#include "CUDAError.h"
#include "Logger.h"

#include <iostream>
#include <stdexcept>

extern "C" void launchLayerNorm(const float* input, float* output, const float* gamma, const float* beta, int rows, int cols, float eps);

LayerNorm::LayerNorm(
    int hiddenSize
)
:
size(hiddenSize),
gamma({hiddenSize}, DataType::FP32),
beta({hiddenSize}, DataType::FP32)

{
    gamma.allocateCPU(); gamma.allocateGPU();
    beta.allocateCPU(); beta.allocateGPU();

    // initialize gamma=1, beta=0
    for(size_t i=0;i<static_cast<size_t>(hiddenSize);i++)
        gamma.cpu()[i] = 1.0f;
    for(size_t i=0;i<static_cast<size_t>(hiddenSize);i++)
        beta.cpu()[i] = 0.0f;
    gamma.upload();
    beta.upload();

    std::cout << "LayerNorm created " << hiddenSize << std::endl;
}


void LayerNorm::forward(
    Tensor& input,
    Tensor& output
)
{
    if(input.rank() != 2 && input.rank() != 3)
        throw std::runtime_error("LayerNorm::forward expected rank 2 or 3");

    int batch = 1;
    int seq = 1;
    int hidden = 0;

    if(input.rank() == 2)
    {
        batch = input.shape()[0];
        hidden = input.shape()[1];
    }
    else
    {
        batch = input.shape()[0];
        seq = input.shape()[1];
        hidden = input.shape()[2];
    }

    if(hidden != size)
        throw std::runtime_error("LayerNorm hidden size mismatch");

    int rows = batch * seq;
    int cols = hidden;

    LOG_INFO_STREAM("LayerNorm::forward rows=" << rows << " cols=" << cols);

    output.allocateGPU();

    launchLayerNorm(input.gpu(), output.gpu(), gamma.gpu(), beta.gpu(), rows, cols, 1e-5f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
