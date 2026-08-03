#include <cuda_runtime.h>
#include "CUDAError.h"

// Launch LayerNorm over rows x cols tensor (row = token, col = hidden)
// gamma and beta are length cols (hidden)

__global__
void layerNormKernel(const float* input, float* output, const float* gamma, const float* beta, int cols, float eps)
{
    int row = blockIdx.x; // one block per row
    const float* rowPtr = input + (size_t)row * cols;
    float* outPtr = output + (size_t)row * cols;

    extern __shared__ float sdata[]; // size = blockDim.x
    float* ssum = sdata;
    float* ssum2 = sdata + blockDim.x;

    int tid = threadIdx.x;
    // compute partial sums
    float sum = 0.0f;
    for(int c = tid; c < cols; c += blockDim.x)
        sum += rowPtr[c];
    ssum[tid] = sum;
    __syncthreads();

    // reduce
    for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
    {
        if(tid < offset) ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }
    float mean = ssum[0] / cols;

    // compute variance
    float sumsq = 0.0f;
    for(int c = tid; c < cols; c += blockDim.x)
    {
        float v = rowPtr[c] - mean;
        sumsq += v * v;
    }
    ssum2[tid] = sumsq;
    __syncthreads();
    for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
    {
        if(tid < offset) ssum2[tid] += ssum2[tid + offset];
        __syncthreads();
    }
    float var = ssum2[0] / cols;
    float denom = rsqrtf(var + eps);

    // normalize
    for(int c = tid; c < cols; c += blockDim.x)
    {
        float norm = (rowPtr[c] - mean) * denom;
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;
        outPtr[c] = norm * g + b;
    }
}

extern "C" void launchLayerNorm(const float* input, float* output, const float* gamma, const float* beta, int rows, int cols, float eps)
{
    int threads = 256;
    if(threads > cols) threads = 128; // small adjustment
    dim3 grid(rows);
    size_t shared = sizeof(float) * threads * 2;
    layerNormKernel<<<grid, threads, shared>>>(input, output, gamma, beta, cols, eps);
    CUDA_CHECK(cudaGetLastError());
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
