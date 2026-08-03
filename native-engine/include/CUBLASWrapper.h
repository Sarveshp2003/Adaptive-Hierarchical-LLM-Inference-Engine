#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "CUDAError.h"
#include <iostream>
#include <stdexcept>

class CUBLASContext
{
public:
    static CUBLASContext& instance();

    void initialize();
    void shutdown();

    // Compute C = A * B^T where A is M x K, B is N x K (row-major), result C is M x N (row-major)
    // Pointers are device pointers.
    void gemmA_Bt(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0);

    // Compute C = A * B where A is M x K, B is K x N (row-major). Device pointers.
    void gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0);

    // Strided-batched versions: matrices stored consecutively per-batch in row-major order.
    // strideA/strideB/strideC given in elements (not bytes). batchCount is number of matrices.
    void gemmA_BtStridedBatched(const float* A, const float* B, float* C, int M, int N, int K,
                                long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream = 0);

    void gemmStridedBatched(const float* A, const float* B, float* C, int M, int N, int K,
                             long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream = 0);

    // Column-major variants: expect each matrix stored column-major contiguous with strides in elements
    void gemmA_BtStridedBatchedCM(const float* A_cm, const float* B_cm, float* C_cm, int M, int N, int K,
                                   long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream = 0);

    void gemmStridedBatchedCM(const float* A_cm, const float* B_cm, float* C_cm, int M, int N, int K,
                               long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream = 0);

private:
    CUBLASContext();
    ~CUBLASContext();

    // helper
    static void cublasCheck(cublasStatus_t status);

    cublasHandle_t handle_;
    bool initialized_;
};

#endif