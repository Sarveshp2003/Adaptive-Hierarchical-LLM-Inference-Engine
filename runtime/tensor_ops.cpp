#include "tensor_ops.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

#if defined(__AVX2__)
inline void gemm_accumulate_avx2(float a_val, const float* b_row, float* out_row, size_t len) {
    size_t j = 0;
    __m256 va = _mm256_set1_ps(a_val);
    for (; j + 7 < len; j += 8) {
        __m256 vb = _mm256_loadu_ps(b_row + j);
        __m256 vout = _mm256_loadu_ps(out_row + j);
        __m256 prod = _mm256_mul_ps(va, vb);
        vout = _mm256_add_ps(vout, prod);
        _mm256_storeu_ps(out_row + j, vout);
    }
    for (; j < len; ++j) out_row[j] += a_val * b_row[j];
}
#endif

} // anonymous namespace

namespace runtime {

std::vector<float> TensorOps::add(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("TensorOps::add size mismatch");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

// Simple dispatch: use parallel tiled for larger matrices
std::vector<float> TensorOps::matmul(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols) {
    // Heuristic: if product size large, use parallel tiled
    const size_t threshold = 256 * 256; // arbitrary
    if (rows * cols >= threshold) {
        return matmul_parallel(a, b, rows, inner, cols, 0);
    }

    if (a.size() != rows * inner || b.size() != inner * cols) {
        throw std::runtime_error("TensorOps::matmul shape mismatch");
    }
    std::vector<float> out(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < inner; ++k) {
                sum += a[r * inner + k] * b[k * cols + c];
            }
            out[r * cols + c] = sum;
        }
    }
    return out;
}

std::vector<float> TensorOps::matmul_tiled(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols, size_t tile) {
    if (a.size() != rows * inner || b.size() != inner * cols) {
        throw std::runtime_error("TensorOps::matmul_tiled shape mismatch");
    }
    std::vector<float> out(rows * cols, 0.0f);
    const size_t BS = tile > 0 ? tile : 32;

    for (size_t ii = 0; ii < rows; ii += BS) {
        size_t i_max = std::min(ii + BS, rows);
        for (size_t kk = 0; kk < inner; kk += BS) {
            size_t k_max = std::min(kk + BS, inner);
            for (size_t jj = 0; jj < cols; jj += BS) {
                size_t j_max = std::min(jj + BS, cols);

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        float a_val = a[i * inner + k];
                        const float *b_row = &b[k * cols + jj];
                        float *out_row = &out[i * cols + jj];
                        size_t len = j_max - jj;
#if defined(__AVX2__)
                        gemm_accumulate_avx2(a_val, b_row, out_row, len);
#else
                        for (size_t j = jj; j < j_max; ++j) {
                            out_row[j - jj] += a_val * b_row[j - jj];
                        }
#endif
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> TensorOps::matmul_parallel(const std::vector<float> &a, const std::vector<float> &b, size_t rows, size_t inner, size_t cols, size_t num_threads) {
    if (a.size() != rows * inner || b.size() != inner * cols) {
        throw std::runtime_error("TensorOps::matmul_parallel shape mismatch");
    }
    if (rows == 0 || cols == 0) return {};

    unsigned int hw = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = hw > 0 ? hw : 2;
    num_threads = std::max<size_t>(1, std::min<size_t>(num_threads, rows));

    std::vector<float> out(rows * cols, 0.0f);
    std::vector<std::thread> threads;
    size_t rows_per = (rows + num_threads - 1) / num_threads;
    const size_t tile = 64;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t r0 = t * rows_per;
        size_t r1 = std::min(rows, r0 + rows_per);
        if (r0 >= r1) break;
        threads.emplace_back([&, r0, r1]() {
            // compute block rows [r0, r1)
            for (size_t ii = r0; ii < r1; ii += tile) {
                size_t i_max = std::min(ii + tile, r1);
                for (size_t kk = 0; kk < inner; kk += tile) {
                    size_t k_max = std::min(kk + tile, inner);
                    for (size_t jj = 0; jj < cols; jj += tile) {
                        size_t j_max = std::min(jj + tile, cols);
                        for (size_t i = ii; i < i_max; ++i) {
                            for (size_t k = kk; k < k_max; ++k) {
                                float a_val = a[i * inner + k];
                                const float *b_row = &b[k * cols + jj];
                                float *out_row = &out[i * cols + jj];
                                for (size_t j = jj; j < j_max; ++j) {
                                    out_row[j - jj] += a_val * b_row[j - jj];
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    for (auto &th : threads) th.join();
    return out;
}

std::vector<float> TensorOps::relu(const std::vector<float> &values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = values[i] > 0.0f ? values[i] : 0.0f;
    return out;
}

// ... rest unchanged

std::vector<float> TensorOps::softmax(const std::vector<float> &logits, size_t rows, size_t cols) {
    if (rows == 0) throw std::invalid_argument("softmax: rows must be > 0");
    if (cols == 0) {
        if (logits.size() % rows != 0) throw std::invalid_argument("softmax: cannot infer cols");
        cols = logits.size() / rows;
    }
    if (logits.size() != rows * cols) throw std::invalid_argument("softmax: shape mismatch");

    std::vector<float> out(logits.size());
    for (size_t r = 0; r < rows; ++r) {
        const float *row_ptr = logits.data() + r * cols;
        float maxv = row_ptr[0];
        for (size_t c = 1; c < cols; ++c) maxv = std::max(maxv, row_ptr[c]);
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = std::exp(row_ptr[c] - maxv);
            sum += out[r * cols + c];
        }
        if (sum == 0.0f) sum = 1e-6f;
        for (size_t c = 0; c < cols; ++c) out[r * cols + c] /= sum;
    }
    return out;
}

std::vector<float> TensorOps::attention(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V, size_t seq_len, size_t dim) {
    if (Q.size() != seq_len * dim || K.size() != seq_len * dim || V.size() != seq_len * dim) {
        throw std::invalid_argument("attention: shape mismatch");
    }

    std::vector<float> out(seq_len * dim, 0.0f);
    std::vector<float> scores(seq_len);
    const float scale = 1.0f / std::sqrt((float)dim);

    for (size_t i = 0; i < seq_len; ++i) {
        // compute scores for query i against all keys
        for (size_t j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            const float *qi = Q.data() + i * dim;
            const float *kj = K.data() + j * dim;
            for (size_t d = 0; d < dim; ++d) dot += qi[d] * kj[d];
            scores[j] = dot * scale;
        }
        // softmax over scores
        float maxv = scores[0];
        for (size_t j = 1; j < seq_len; ++j) maxv = std::max(maxv, scores[j]);
        float ssum = 0.0f;
        for (size_t j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - maxv);
            ssum += scores[j];
        }
        if (ssum == 0.0f) ssum = 1e-6f;
        for (size_t j = 0; j < seq_len; ++j) scores[j] /= ssum;

        // weighted sum over V
        for (size_t d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                acc += scores[j] * V[j * dim + d];
            }
            out[i * dim + d] = acc;
        }
    }

    return out;
}

std::vector<float> TensorOps::add_ptr(const std::vector<float> &a, const float *b_ptr, size_t b_count) {
    if (!b_ptr) throw std::invalid_argument("b_ptr is null");
    size_t n = std::min(a.size(), b_count);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b_ptr[i];
    return out;
}

std::vector<float> TensorOps::matmul_ptr(const std::vector<float> &a, const float *b_ptr, size_t rows, size_t inner, size_t cols) {
    if (!b_ptr) throw std::invalid_argument("b_ptr is null");
    if (a.size() != rows * inner) throw std::runtime_error("TensorOps::matmul_ptr shape mismatch for a");

    std::vector<float> out(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t k = 0; k < inner; ++k) {
            float av = a[r * inner + k];
            const float *b_row = b_ptr + k * cols;
            for (size_t c = 0; c < cols; ++c) {
                out[r * cols + c] += av * b_row[c];
            }
        }
    }
    return out;
}

}
