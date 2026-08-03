#include "cuda_executor.h"
#include "../tensor_ops.h"
#include <iostream>
#include <string>
#include <vector>
#include <Windows.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace runtime { namespace gpu {

bool cuda_available() {
    const char *path_env = std::getenv("PATH");
    if (!path_env) return false;
    std::string pathstr(path_env);
    size_t start = 0;
    while (start < pathstr.size()) {
        size_t pos = pathstr.find(';', start);
        if (pos == std::string::npos) pos = pathstr.size();
        std::string part = pathstr.substr(start, pos - start);
        if (!part.empty()) {
            // Build search pattern: <part>\\cudart64_*.dll
            std::wstring wpart(part.begin(), part.end());
            if (!wpart.empty()) {
                std::wstring pattern = wpart;
                if (pattern.back() != L'\\' && pattern.back() != L'/') pattern += L"\\";
                pattern += L"cudart64_*.dll";
                WIN32_FIND_DATAW findData;
                HANDLE hFind = FindFirstFileW(pattern.c_str(), &findData);
                if (hFind != INVALID_HANDLE_VALUE) {
                    do {
                        std::wstring filePath = wpart;
                        if (filePath.back() != L'\\' && filePath.back() != L'/') filePath += L"\\";
                        filePath += findData.cFileName;
                        HMODULE h = LoadLibraryW(filePath.c_str());
                        if (!h) continue;
                        typedef int (__stdcall *cudaGetDeviceCount_t)(int*);
                        auto pCount = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
                        if (!pCount) { FreeLibrary(h); continue; }
                        int dev = 0;
                        int err = pCount(&dev);
                        FreeLibrary(h);
                        if (err == 0 && dev > 0) { FindClose(hFind); return true; }
                    } while (FindNextFileW(hFind, &findData));
                    FindClose(hFind);
                }
            }
        }
        start = pos + 1;
    }
    return false;
}

bool cuda_init() {
    // No real CUDA in this stub; report simulated availability
    std::cerr << "[gpu::cuda_executor] CUDA stub initialized (simulated)\n";
    return true;
}

bool cuda_gemm(const float *A, const float *B, float *C, size_t rows, size_t inner, size_t cols) {
    if (!A || !B || !C) return false;
    // fallback to CPU implementation using TensorOps::matmul_ptr
    std::vector<float> a_vec(A, A + rows * inner);
    std::vector<float> result = runtime::TensorOps::matmul_ptr(a_vec, B, rows, inner, cols);
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

// Provide lightweight device-weight cache and GEMM variant for linkage when CUDA .cu not linked
#include <unordered_map>
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
    // Treat dB as host pointer in this stub and perform CPU matmul
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
