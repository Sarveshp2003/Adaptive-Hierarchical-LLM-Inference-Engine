#include "FeedForward.h"
#include "TensorOps.h"
#include "CUBLASWrapper.h"
#include "CUDAStream.h"

#include <iostream>
#include <stdexcept>

FeedForward::FeedForward(int hiddenSize)
:
hiddenSize(hiddenSize),
fc1(hiddenSize, hiddenSize * 4),
fc2(hiddenSize * 4, hiddenSize),
hidden({1,1}, DataType::FP32),
initialized(false)
{
    std::cout << "FeedForward created: " << hiddenSize << " -> " << hiddenSize * 4 << " -> " << hiddenSize << std::endl;
}

void FeedForward::forward(Tensor& input, Tensor& output)
{
    if(input.rank() != 2 && input.rank() != 3)
    {
        throw std::runtime_error("FeedForward::forward expected rank 2 or 3");
    }

    int batch = 1;
    int seq = 1;
    int inHidden = 0;

    if(input.rank() == 2)
    {
        batch = input.shape()[0];
        inHidden = input.shape()[1];
    }
    else
    {
        batch = input.shape()[0];
        seq = input.shape()[1];
        inHidden = input.shape()[2];
    }

    if(inHidden != hiddenSize)
    {
        throw std::runtime_error("FeedForward::forward hidden size mismatch");
    }

    // Initialize hidden buffer to match input batch/sequence on first use
    std::vector<int> desiredShape = { batch, seq, hiddenSize * 4 };

    if(!initialized || hidden.shape().size() != 3 || hidden.shape()[0] != desiredShape[0] || hidden.shape()[1] != desiredShape[1] || hidden.shape()[2] != desiredShape[2])
    {
        hidden = Tensor(desiredShape, DataType::FP32);
        // allocate only GPU memory for the intermediate buffer
        hidden.allocateGPU();
        initialized = true;
    }

    // Fused FFN using cuBLAS + GELU kernel to avoid CPU round trips
    int rows = batch * seq;
    int hiddenFF = hiddenSize * 4;
    int outHidden = hiddenSize;

    cudaStream_t stream = CUDAStream::get();

    // Compute FC1: hidden = input * W1  (rows x inHidden) * (inHidden x hiddenFF) -> rows x hiddenFF
    CUBLASContext::instance().gemm(input.gpu(), fc1.getWeights().gpu(), hidden.gpu(), rows, hiddenFF, inHidden, stream);
    CUDA_CHECK(cudaGetLastError());

    // Apply GELU in-place using high-level TensorOps::gelu wrapper
    gelu(hidden, hidden);
    CUDA_CHECK(cudaGetLastError());

    // Compute FC2: output = hidden * W2  (rows x hiddenFF) * (hiddenFF x outHidden) -> rows x outHidden
    output.allocateGPU();
    CUBLASContext::instance().gemm(hidden.gpu(), fc2.getWeights().gpu(), output.gpu(), rows, outHidden, hiddenFF, stream);
    CUDA_CHECK(cudaGetLastError());

    // synchronize to make sure result is ready
    CUDA_CHECK(cudaDeviceSynchronize());
}
