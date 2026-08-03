#include <cuda_runtime.h>
#include <cstdio>

// Row-major (rows x cols) -> Column-major (rows x cols) i.e., dst[i + j*rows] = src[i*cols + j]
__global__ void row_to_colmajor_kernel(const float* src, float* dst, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < rows && c < cols)
    {
        dst[r + c * (size_t)rows] = src[r * (size_t)cols + c];
    }
}

// Row-major (N x K) -> Column-major for transposed shape (K x N): dst[p + q*K] = src[q*K + p]
__global__ void row_to_colmajor_transpose_kernel(const float* src, float* dst, int N, int K)
{
    int q = blockIdx.y * blockDim.y + threadIdx.y; // q in [0,N)
    int p = blockIdx.x * blockDim.x + threadIdx.x; // p in [0,K)
    if(q < N && p < K)
    {
        dst[p + q * (size_t)K] = src[q * (size_t)K + p];
    }
}

// Column-major (rows x cols) -> Row-major (rows x cols): dst[r*cols + c] = src[r + c*rows]
__global__ void col_to_rowmajor_kernel(const float* src, float* dst, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < rows && c < cols)
    {
        dst[r * (size_t)cols + c] = src[r + c * (size_t)rows];
    }
}

extern "C" void launchRowToColMajor(const float* src, float* dst, int rows, int cols)
{
    dim3 block(16,16);
    dim3 grid((cols + block.x - 1)/block.x, (rows + block.y - 1)/block.y);
    row_to_colmajor_kernel<<<grid, block>>>(src, dst, rows, cols);
}

extern "C" void launchRowToColMajorTranspose(const float* src, float* dst, int N, int K)
{
    dim3 block(16,16);
    dim3 grid((K + block.x - 1)/block.x, (N + block.y - 1)/block.y);
    row_to_colmajor_transpose_kernel<<<grid, block>>>(src, dst, N, K);
}

extern "C" void launchColToRowMajor(const float* src, float* dst, int rows, int cols)
{
    dim3 block(16,16);
    dim3 grid((cols + block.x - 1)/block.x, (rows + block.y - 1)/block.y);
    col_to_rowmajor_kernel<<<grid, block>>>(src, dst, rows, cols);
}
