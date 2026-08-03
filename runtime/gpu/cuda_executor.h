#pragma once

#include <cstddef>
#include <vector>

namespace runtime { namespace gpu {

// Minimal CUDA executor stub API. When CUDA is enabled, replace implementations
// with real device kernels. For now these call CPU fallbacks.

bool cuda_available();

// Initialize CUDA runtime (stub)
bool cuda_init();

// GEMM: C = A * B (A rows x inner, B inner x cols)
bool cuda_gemm(const float *A, const float *B, float *C, size_t rows, size_t inner, size_t cols);

// Softmax (row-wise)
bool cuda_softmax(const float *logits, float *out, size_t rows, size_t cols);

// Attention: Q,K,V flattened row-major (seq_len x dim)
bool cuda_attention(const float *Q, const float *K, const float *V, float *out, size_t seq_len, size_t dim);

// Persistent weight device management
bool cuda_alloc_weight(size_t id, const float *host_ptr, size_t float_count);
const float* cuda_get_weight_ptr(size_t id); // returns device pointer for weight id or nullptr
void cuda_free_weight(size_t id);

// GEMM variant that accepts device pointer for B (weights)
bool cuda_gemm_hostA_devB_hostC(const float *A_host, const float *dB, float *C_host, size_t rows, size_t inner, size_t cols);

}} // namespace runtime::gpu
