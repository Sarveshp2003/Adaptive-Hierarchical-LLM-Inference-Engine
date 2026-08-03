#include <cuda_runtime.h>
#include <math.h>
#include "CUDAError.h"

__global__
void softmaxKernel(
    const float* input,
    float* output,
    int rows,
    int cols
)
{
    int row = blockIdx.x;

    if(row >= rows)
        return;

    // Find maximum value for numerical stability
    float maxValue = input[row * cols];

    for(int i = 1; i < cols; i++)
    {
        float value = input[row * cols + i];

        if(value > maxValue)
            maxValue = value;
    }

    // Compute exponentials and sum
    float sum = 0.0f;

    for(int i = 0; i < cols; i++)
    {
        float value = expf(
            input[row * cols + i] - maxValue
        );

        output[row * cols + i] = value;

        sum += value;
    }

    // Normalize
    for(int i = 0; i < cols; i++)
    {
        output[row * cols + i] /= sum;
    }
}

// Scaled softmax: multiplies inputs by scale before computing softmax (useful for attention scaling)
__global__
void scaledSoftmaxKernel(
    const float* input,
    float* output,
    int rows,
    int cols,
    float scale
)
{
    int row = blockIdx.x;

    if(row >= rows)
        return;

    // Find maximum value for numerical stability
    float maxValue = input[row * cols] * scale;

    for(int i = 1; i < cols; i++)
    {
        float value = input[row * cols + i] * scale;

        if(value > maxValue)
            maxValue = value;
    }

    // Compute exponentials and sum
    float sum = 0.0f;

    for(int i = 0; i < cols; i++)
    {
        float value = expf(
            input[row * cols + i] * scale - maxValue
        );

        output[row * cols + i] = value;

        sum += value;
    }

    // Normalize
    for(int i = 0; i < cols; i++)
    {
        output[row * cols + i] /= sum;
    }
}

extern "C"
void launchSoftmax(
    float* input,
    float* output,
    int rows,
    int cols
)
{
    softmaxKernel<<<rows,1>>>(
        input,
        output,
        rows,
        cols
    );

    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

extern "C"
void launchScaledSoftmax(
    float* input,
    float* output,
    int rows,
    int cols,
    float scale
)
{
    scaledSoftmaxKernel<<<rows,1>>>(
        input,
        output,
        rows,
        cols,
        scale
    );

    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}