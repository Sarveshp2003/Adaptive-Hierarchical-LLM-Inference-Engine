#include "cuda_executor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <mutex>

namespace runtime { namespace gpu {

static cublasHandle_t g_cublas = nullptr;
static bool g_initialized = false;

// Simple device buffer pool for reuse
static float *g_buf = nullptr;
static size_t g_buf_bytes = 0;
static std::mutex g_buf_mutex;

static bool ensure_device_buffer(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    if (g_buf_bytes >= bytes && g_buf) return true;
    if (g_buf) {
        cudaFree(g_buf);
        g_buf = nullptr;
        g_buf_bytes = 0;
    }
    cudaError_t cerr = cudaMalloc((void**)&g_buf, bytes);
    if (cerr != cudaSuccess) {
        g_buf = nullptr; g_buf_bytes = 0; return false;
    }
    g_buf_bytes = bytes;
    return true;
}

// Persistent weight storage on device
#include <unordered_map>
static std::unordered_map<size_t, float*> g_weight_dev_map;
static std::mutex g_weight_map_mutex;

bool cuda_alloc_weight(size_t id, const float *host_ptr, size_t float_count) {
    if (!host_ptr || float_count == 0) return false;
    if (!cuda_init()) return false;
    std::lock_guard<std::mutex> lk(g_weight_map_mutex);
    if (g_weight_dev_map.find(id) != g_weight_dev_map.end()) return true; // already allocated
    float *dptr = nullptr;
    size_t bytes = float_count * sizeof(float);
    cudaError_t cerr = cudaMalloc((void**)&dptr, bytes);
    if (cerr != cudaSuccess) return false;
    if (cudaMemcpy(dptr, host_ptr, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(dptr); return false; }
    g_weight_dev_map[id] = dptr;
    return true;
}

const float* cuda_get_weight_ptr(size_t id) {
    std::lock_guard<std::mutex> lk(g_weight_map_mutex);
    auto it = g_weight_dev_map.find(id);
    if (it == g_weight_dev_map.end()) return nullptr;
    return it->second;
}

void cuda_free_weight(size_t id) {
    std::lock_guard<std::mutex> lk(g_weight_map_mutex);
    auto it = g_weight_dev_map.find(id);
    if (it == g_weight_dev_map.end()) return;
    cudaFree(it->second);
    g_weight_dev_map.erase(it);
}

bool cuda_gemm_hostA_devB_hostC(const float *A, const float *dB, float *C, size_t rows, size_t inner, size_t cols) {
    if (!A || !dB || !C) return false;
    if (!cuda_init()) return false;

    // Use pooled buffer for dA and dC
    size_t sizeA = rows * inner * sizeof(float);
    size_t sizeC = rows * cols * sizeof(float);
    if (!ensure_device_buffer(sizeA + sizeC)) return false;
    float *dA = g_buf;
    float *dC = (float*)((char*)g_buf + sizeA);

    if (cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // note: dB is device pointer (inner x cols)
    cublasStatus_t stat = cublasSgemm(g_cublas,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)cols, (int)rows, (int)inner,
                                     &alpha,
                                     dB, (int)inner,
                                     dA, (int)inner,
                                     &beta,
                                     dC, (int)cols);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

bool cuda_available() {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);

    // Write diagnostic to workspace file for reliable capture
    try {
        std::filesystem::path p("E:/adaptivellm/tmp");
        std::filesystem::create_directories(p);
        std::ofstream f((p / "cuda_diag.txt").string(), std::ios::app);
        if (!f) {
            std::cerr << "[cuda_executor] failed to open cuda_diag.txt" << std::endl;
        } else {
            if (err != cudaSuccess) {
                f << "cudaGetDeviceCount error: " << cudaGetErrorString(err) << " (code " << err << ")\n";
            } else {
                f << "cudaGetDeviceCount reports " << devCount << " device(s)\n";
            }
            f.close();
        }
    } catch (...) {
        // best-effort
    }

    if (err != cudaSuccess) {
        std::cerr << "[cuda_executor] cudaGetDeviceCount error: " << cudaGetErrorString(err) << " (code " << err << ")" << std::endl;
        return false;
    }
    std::cerr << "[cuda_executor] cudaGetDeviceCount reports " << devCount << " device(s)" << std::endl;
    return devCount > 0;
}

bool cuda_init() {
    if (g_initialized) return true;
    if (!cuda_available()) return false;
    cudaError_t cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        std::cerr << "[cuda_executor] cudaSetDevice failed: " << cudaGetErrorString(cerr) << "\n";
        return false;
    }
    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuda_executor] cublasCreate failed\n";
        return false;
    }
    g_initialized = true;
    std::cerr << "[cuda_executor] CUDA/cuBLAS initialized\n";
    return true;
}

// GPU softmax kernel (row-wise). blockDim.x threads per row.
__global__ static void softmax_row_kernel(float *scores, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float maxv = -1e30f;
    int base = row * cols;
    for (int c = tid; c < cols; c += stride) {
        float v = scores[base + c];
        if (v > maxv) maxv = v;
    }
    // reduce max
    sdata[tid] = maxv;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    maxv = sdata[0];
    float sum = 0.0f;
    for (int c = tid; c < cols; c += stride) {
        float e = expf(scores[base + c] - maxv);
        sdata[tid] = e; // reuse shared for partial sums
        sum += e;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];
    for (int c = tid; c < cols; c += stride) {
        scores[base + c] = expf(scores[base + c] - maxv) / sum; // recompute expf, simpler correctness
    }
}

bool cuda_gemm(const float *A, const float *B, float *C, size_t rows, size_t inner, size_t cols) {
    if (!A || !B || !C) return false;
    if (!cuda_init()) return false;

    // Allocate device memory (use pool if large)
    size_t sizeA = rows * inner * sizeof(float);
    size_t sizeB = inner * cols * sizeof(float);
    size_t sizeC = rows * cols * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (!ensure_device_buffer(sizeA + sizeB + sizeC)) return false;
    // partition g_buf
    dA = g_buf;
    dB = (float*)((char*)g_buf + sizeA);
    dC = (float*)((char*)g_buf + sizeA + sizeB);

    if (cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t stat = cublasSgemm(g_cublas,
                                     CUBLAS_OP_T, CUBLAS_OP_T,
                                     (int)cols, (int)rows, (int)inner,
                                     &alpha,
                                     dB, (int)inner,
                                     dA, (int)rows,
                                     &beta,
                                     dC, (int)cols);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuda_executor] cublasSgemm failed\n";
        return false;
    }

    if (cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

bool cuda_softmax(const float *logits, float *out, size_t rows, size_t cols) {
    if (!logits || !out) return false;
    if (!cuda_init()) return false;

    size_t size = rows * cols * sizeof(float);
    if (!ensure_device_buffer(size)) return false;
    float *d = g_buf;
    if (cudaMemcpy(d, logits, size, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    int block = 256;
    int grid = (int)rows;
    int shared = block * sizeof(float);
    softmax_row_kernel<<<grid, block, shared>>>(d, (int)rows, (int)cols);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::cerr << "[cuda_executor] softmax kernel launch failed: " << cudaGetErrorString(cerr) << "\n";
        return false;
    }
    if (cudaMemcpy(out, d, size, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

bool cuda_attention(const float *Q, const float *K, const float *V, float *out, size_t seq_len, size_t dim) {
    if (!Q || !K || !V || !out) return false;
    if (!cuda_init()) return false;

    // Allocate device buffers: Q (seq*dim), K (seq*dim), scores (seq*seq), V (seq*dim), out (seq*dim)
    size_t qsize = seq_len * dim * sizeof(float);
    size_t ksize = qsize;
    size_t vsize = qsize;
    size_t scores_size = seq_len * seq_len * sizeof(float);
    size_t out_size = qsize;
    size_t total = qsize + ksize + scores_size + vsize + out_size;
    if (!ensure_device_buffer(total)) return false;
    char *base = (char*)g_buf;
    float *dQ = (float*)base;
    float *dK = (float*)(base + qsize);
    float *dScores = (float*)(base + qsize + ksize);
    float *dV = (float*)(base + qsize + ksize + scores_size);
    float *dOut = (float*)(base + qsize + ksize + scores_size + vsize);

    if (cudaMemcpy(dQ, Q, qsize, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(dK, K, ksize, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(dV, V, vsize, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    // scores = Q * K^T  (seq x seq)
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t stat = cublasSgemm(g_cublas,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)seq_len, (int)seq_len, (int)dim,
                                     &alpha,
                                     dK, (int)dim,
                                     dQ, (int)dim,
                                     &beta,
                                     dScores, (int)seq_len);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuda_executor] cublasSgemm for scores failed\n";
        return false;
    }

    // softmax across rows of scores (in-place)
    int block = 256;
    int grid = (int)seq_len;
    int shared = block * sizeof(float);
    softmax_row_kernel<<<grid, block, shared>>>(dScores, (int)seq_len, (int)seq_len);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::cerr << "[cuda_executor] softmax kernel launch failed: " << cudaGetErrorString(cerr) << "\n";
        return false;
    }

    // out = scores * V  (seq x dim)
    stat = cublasSgemm(g_cublas,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       (int)dim, (int)seq_len, (int)seq_len,
                       &alpha,
                       dV, (int)dim,
                       dScores, (int)seq_len,
                       &beta,
                       dOut, (int)dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuda_executor] cublasSgemm for out failed\n";
        return false;
    }

    if (cudaMemcpy(out, dOut, out_size, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

}} // namespace runtime::gpu
