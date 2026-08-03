#include <cuda_runtime.h>

__global__
void gatherHeadKernel(
    const float* src,
    float* dst,
    int rows,
    int headDim,
    int headOffset,
    int stride
)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < rows && c < headDim)
    {
        dst[r * headDim + c] = src[r * stride + headOffset + c];
    }
}

extern "C"
void launchGatherHead(
    const float* src,
    float* dst,
    int rows,
    int headDim,
    int headOffset,
    int stride
)
{
    dim3 threads(16,16);
    dim3 blocks((headDim + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    gatherHeadKernel<<<blocks, threads>>>(src, dst, rows, headDim, headOffset, stride);
}
