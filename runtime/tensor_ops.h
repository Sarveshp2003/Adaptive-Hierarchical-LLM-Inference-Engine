#pragma once

#include <vector>
#include <cstddef>

namespace runtime {

struct TensorOps {
    static std::vector<float> add(const std::vector<float> &a, const std::vector<float> &b);
    static std::vector<float> matmul(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols);
    // Tiled/blocked GEMM for larger matrices
    static std::vector<float> matmul_tiled(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols, size_t tile);
    // Multithreaded GEMM
    static std::vector<float> matmul_parallel(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols, size_t num_threads = 0);
    static std::vector<float> relu(const std::vector<float> &values);

    // Row-wise softmax: logits shaped rows x cols -> same shape
    static std::vector<float> softmax(const std::vector<float> &logits, size_t rows, size_t cols);

    // Attention: Q,K,V are seq_len x dim (flattened row-major). Returns output seq_len x dim.
    static std::vector<float> attention(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V, size_t seq_len, size_t dim);

    // Zero-copy variants: b provided as pointer to float array
    static std::vector<float> add_ptr(const std::vector<float> &a, const float *b_ptr, size_t b_count);
    static std::vector<float> matmul_ptr(const std::vector<float> &a, const float *b_ptr, size_t rows, size_t inner, size_t cols);
};

}
