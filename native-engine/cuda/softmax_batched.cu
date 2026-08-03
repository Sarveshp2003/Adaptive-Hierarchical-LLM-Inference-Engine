#include <cuda_runtime.h>
#include <math.h>

// Parallel per-row softmax using one block per row. Assumes cols <= 1024 for now.
extern "C" __global__ void softmax_batched_kernel(float* data, int totalRows, int cols, float scale)
{
    int row = blockIdx.x;
    if(row >= totalRows) return;

    extern __shared__ float sdata[]; // dynamic shared memory
    float* smax = sdata; // first element store per-block max partials

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int offset = row * cols;

    // compute max in parallel
    float local_max = -1e20f;
    for(int c = tid; c < cols; c += stride)
    {
        float v = data[offset + c] * scale;
        if(v > local_max) local_max = v;
    }

    // reduction for max
    smax[tid] = local_max;
    __syncthreads();

    // tree reduction
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            float a = smax[tid];
            float b = smax[tid + s];
            if(b > a) smax[tid] = b;
        }
        __syncthreads();
    }

    float row_max = smax[0];

    // compute exponentials and sum in parallel
    float local_sum = 0.0f;
    for(int c = tid; c < cols; c += stride)
    {
        float e = expf(data[offset + c] * scale - row_max);
        data[offset + c] = e;
        local_sum += e;
    }

    // reduction for sum
    smax[tid] = local_sum;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            smax[tid] += smax[tid + s];
        __syncthreads();
    }

    float row_sum = smax[0];
    if(row_sum == 0.0f) return;

    // normalize
    for(int c = tid; c < cols; c += stride)
    {
        data[offset + c] = data[offset + c] / row_sum;
    }
}

extern "C" void launchScaledSoftmaxBatched(float* data, int batchCount, int rows, int cols, float scale)
{
    int total = batchCount * rows;
    int threads = min(256, cols);
    int blocks = total;
    int shared = threads * sizeof(float);
    softmax_batched_kernel<<<blocks, threads, shared>>>(data, total, cols, scale);
    cudaDeviceSynchronize();
}
