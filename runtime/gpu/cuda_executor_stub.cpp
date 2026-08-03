#include "cuda_executor.h"
#include "../tensor_ops.h"
#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <iostream>

namespace runtime { namespace gpu {

bool cuda_available() { return false; }

bool cuda_init() { std::cerr << "[gpu::cuda_executor_stub] CUDA disabled - using CPU fallback\n"; return false; }

bool cuda_gemm(const float *A, const float *B, float *C, size_t rows, size_t inner, size_t cols) {
    if (!A || !B || !C) return false;
    std::vector<float> a_vec(A, A + rows * inner);
    auto result = runtime::TensorOps::matmul_ptr(a_vec, B, rows, inner, cols);
    std::copy(result.begin(), result.end(), C);
    return true;
}

bool cuda_softmax(const float *logits, float *out, size_t rows, size_t cols) {
    if (!logits || !out) return false;
    std::vector<float> in(logits, logits + rows * cols);
    auto r = runtime::TensorOps::softmax(in, rows, cols);
    std::copy(r.begin(), r.end(), out);
    return true;
}

bool cuda_attention(const float *Q, const float *K, const float *V, float *out, size_t seq_len, size_t dim) {
    if (!Q || !K || !V || !out) return false;
    std::vector<float> q(Q, Q + seq_len * dim);
    std::vector<float> k(K, K + seq_len * dim);
    std::vector<float> v(V, V + seq_len * dim);
    auto r = runtime::TensorOps::attention(q,k,v,seq_len,dim);
    std::copy(r.begin(), r.end(), out);
    return true;
}

static std::unordered_map<size_t, float*> g_weight_dev_map_stub;
static std::mutex g_weight_map_mutex_stub;

bool cuda_alloc_weight(size_t id, const float *host_ptr, size_t float_count) {
    if (!host_ptr || float_count == 0) return false;
    float *dptr = new (std::nothrow) float[float_count];
    if (!dptr) return false;
    std::memcpy(dptr, host_ptr, float_count * sizeof(float));
    std::lock_guard<std::mutex> lk(g_weight_map_mutex_stub);
    g_weight_dev_map_stub[id] = dptr;
    return true;
}

const float* cuda_get_weight_ptr(size_t id) {
    std::lock_guard<std::mutex> lk(g_weight_map_mutex_stub);
    auto it = g_weight_dev_map_stub.find(id);
    if (it == g_weight_dev_map_stub.end()) return nullptr;
    return it->second;
}

void cuda_free_weight(size_t id) {
    std::lock_guard<std::mutex> lk(g_weight_map_mutex_stub);
    auto it = g_weight_dev_map_stub.find(id);
    if (it == g_weight_dev_map_stub.end()) return;
    delete [] it->second;
    g_weight_dev_map_stub.erase(it);
}

bool cuda_gemm_hostA_devB_hostC(const float *A_host, const float *dB, float *C_host, size_t rows, size_t inner, size_t cols) {
    if (!A_host || !dB || !C_host) return false;
    const float *B_host = dB;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (size_t k = 0; k < inner; ++k) acc += A_host[r*inner + k] * B_host[k*cols + c];
            C_host[r*cols + c] = acc;
        }
    }
    return true;
}

}} // namespace runtime::gpu
