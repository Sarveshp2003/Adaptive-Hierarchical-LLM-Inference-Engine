#include "CUBLASWrapper.h"
#include "CUDAError.h"
#include "AllocationTracker.h"
#include "RuntimeMemory.h"
#include "Logger.h"

#include <iostream>
#include <stdexcept>

// Force use of tracked cudaMalloc for large cuBLAS scratch temporaries to rule out pool corruption
#define FORCE_TRACKED_SCRATCH 1

// transpose helpers in cuda/transpose.cu
extern "C" void launchRowToColMajor(const float* src, float* dst, int rows, int cols);
extern "C" void launchRowToColMajorTranspose(const float* src, float* dst, int N, int K);
extern "C" void launchColToRowMajor(const float* src, float* dst, int rows, int cols);

CUBLASContext::CUBLASContext()
: handle_(nullptr), initialized_(false)
{
}

CUBLASContext::~CUBLASContext()
{
    shutdown();
}

CUBLASContext& CUBLASContext::instance()
{
    static CUBLASContext ctx;
    return ctx;
}

static bool performDeviceSanityCheck()
{
    const size_t testSize = 4096; // small test buffer
    void* dptr = nullptr;
    cudaError_t e = AllocationTracker::trackedCudaMalloc(&dptr, testSize, "performDeviceSanityCheck");
    if(e != cudaSuccess || dptr == nullptr) {
        std::cerr << "performDeviceSanityCheck: trackedCudaMalloc failed: " << cudaGetErrorString(e) << std::endl;
        return false;
    }
    // fill with pattern
    const uint32_t pattern = 0xA5A5A5A5;
    e = cudaMemset(dptr, 0xA5, testSize);
    if(e != cudaSuccess) {
        std::cerr << "performDeviceSanityCheck: cudaMemset failed: " << cudaGetErrorString(e) << std::endl;
        AllocationTracker::trackedCudaFree(dptr, "performDeviceSanityCheck cleanup");
        return false;
    }
    // sync and copy back small sample
    e = cudaDeviceSynchronize();
    if(e != cudaSuccess) {
        std::cerr << "performDeviceSanityCheck: cudaDeviceSynchronize failed: " << cudaGetErrorString(e) << std::endl;
        AllocationTracker::trackedCudaFree(dptr, "performDeviceSanityCheck cleanup");
        return false;
    }
    uint8_t hostSample[16] = {0};
    e = cudaMemcpy(hostSample, dptr, sizeof(hostSample), cudaMemcpyDeviceToHost);
    if(e != cudaSuccess) {
        std::cerr << "performDeviceSanityCheck: cudaMemcpy D2H failed: " << cudaGetErrorString(e) << std::endl;
        AllocationTracker::trackedCudaFree(dptr, "performDeviceSanityCheck cleanup");
        return false;
    }
    // quick check that bytes equal 0xA5
    for(size_t i=0;i<sizeof(hostSample);++i) {
        if(hostSample[i] != 0xA5) {
            std::cerr << "performDeviceSanityCheck: sample mismatch at " << i << " val=" << (int)hostSample[i] << std::endl;
            AllocationTracker::trackedCudaFree(dptr, "performDeviceSanityCheck cleanup");
            return false;
        }
    }
    AllocationTracker::trackedCudaFree(dptr, "performDeviceSanityCheck cleanup");
    return true;
}

void CUBLASContext::initialize()
{
    if(initialized_) return;
    // perform a quick device-side sanity check before creating cuBLAS context to detect device memory corruption
    if(!performDeviceSanityCheck()) {
        std::cerr << "Device sanity check FAILED prior to cublasCreate. Serializing allocator state and aborting." << std::endl;
        try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_before_cublas.json")); } catch(...) {}
        throw std::runtime_error("Device sanity check failed before initializing cuBLAS");
    }
    cublasStatus_t s = cublasCreate(&handle_);
    cublasCheck(s);
    // post-init check
    if(!performDeviceSanityCheck()) {
        std::cerr << "Device sanity check FAILED after cublasCreate. Serializing allocator state." << std::endl;
        try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_after_cublas.json")); } catch(...) {}
    }
    initialized_ = true;
}

void CUBLASContext::shutdown()
{
    if(!initialized_) return;
    cublasCheck(cublasDestroy(handle_));
    handle_ = nullptr;
    initialized_ = false;
}

static const char* cublasStatusName(cublasStatus_t s)
{
    switch(s)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

void CUBLASContext::cublasCheck(cublasStatus_t status)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error: " << (int)status << " (" << cublasStatusName(status) << ")" << std::endl;
        throw std::runtime_error("cuBLAS call failed");
    }
}

static void validateDevicePointer(const void* ptr, const char* name)
{
    if(ptr == nullptr)
    {
        std::cerr << "Device pointer " << name << " is null" << std::endl;
        throw std::runtime_error("Null device pointer");
    }
    cudaPointerAttributes attr;
    cudaError_t cerr = cudaPointerGetAttributes(&attr, ptr);
    if(cerr != cudaSuccess)
    {
        std::cerr << "cudaPointerGetAttributes failed for " << name << ": " << cerr << std::endl;
    }
    else
    {
        // Print basic info for debugging
        std::cerr << "Ptr " << name << " attrs: device = " << attr.device << " type = " << attr.type << std::endl;
    }
}

void CUBLASContext::gemmA_Bt(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    if(!initialized_) initialize();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if(stream != 0)
    {
        cublasCheck(cublasSetStream(handle_, stream));
    }

    // C (M x N) = A (M x K) * B^T (K x N) -- our tensors are row-major.
    // Use row-major -> column-major transpose trick: compute C^T = B * A^T using cuBLAS column-major
    // Prefer robust column-major transpose fallback to avoid illegal parameter errors in cublas mappings.
    std::cerr << "cuBLAS gemmA_Bt: using transpose-conversion fallback (robust path)" << std::endl;

    float* A_cm = nullptr; // M x K col-major
    float* B_cm = nullptr; // K x N col-major (transposed layout)
    float* C_cm = nullptr; // M x N col-major

    bool A_fromPool = false, B_fromPool = false, C_fromPool = false;

    size_t A_cm_bytes = (size_t)M * K * sizeof(float);
    size_t B_cm_bytes = (size_t)K * N * sizeof(float);
    size_t C_cm_bytes = (size_t)M * N * sizeof(float);

    // Try pooled GPU allocation first, prefer scratch partition for temporaries, then fallback to tracked cudaMalloc
#ifndef FORCE_TRACKED_SCRATCH
    void* tmp = RuntimeMemory::allocateScratchGPU(A_cm_bytes);
#else
    void* tmp = nullptr;
#endif
    if(tmp) { A_cm = (float*)tmp; A_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&A_cm, A_cm_bytes, "CUBLASWrapper::gemmA_Bt A_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc A_cm failed: " << cerr << std::endl; cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(B_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { B_cm = (float*)tmp; B_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&B_cm, B_cm_bytes, "CUBLASWrapper::gemmA_Bt B_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc B_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt cleanup"); cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(C_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { C_cm = (float*)tmp; C_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&C_cm, C_cm_bytes, "CUBLASWrapper::gemmA_Bt C_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc C_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt cleanup"); if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_Bt cleanup"); cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

    // Log pointer attributes for diagnostics
    cudaPointerAttributes aattr, battr, cattr;
    if(A_cm) { cudaPointerGetAttributes(&aattr, A_cm); LOG_INFO_STREAM("A_cm ptr="<< (void*)A_cm << " attrs.type=" << aattr.type << " device=" << aattr.device); }
    if(B_cm) { cudaPointerGetAttributes(&battr, B_cm); LOG_INFO_STREAM("B_cm ptr="<< (void*)B_cm << " attrs.type=" << battr.type << " device=" << battr.device); }
    if(C_cm) { cudaPointerGetAttributes(&cattr, C_cm); LOG_INFO_STREAM("C_cm ptr="<< (void*)C_cm << " attrs.type=" << cattr.type << " device=" << cattr.device); }

    // A is row-major M x K -> A_cm column-major M x K
    launchRowToColMajor(A, A_cm, M, K);
    // B is row-major N x K -> B_cm should be K x N column-major (transpose)
    launchRowToColMajorTranspose(B, B_cm, N, K);

    // Perform sgemm on column-major buffers: C_cm = A_cm (M x K) * B_cm (K x N)
    LOG_INFO_STREAM("cublasSgemm about to run (gemmA_Bt). M=" << M << " N=" << N << " K=" << K << " A_cm=" << (void*)A_cm << " B_cm=" << (void*)B_cm << " C_cm=" << (void*)C_cm << " A_fromPool=" << A_fromPool << " B_fromPool=" << B_fromPool << " C_fromPool=" << C_fromPool);
    // device sanity check before calling cuBLAS
    if(!performDeviceSanityCheck()) {
        std::cerr << "Device sanity check FAILED before cublasSgemm (gemmA_Bt). Serializing allocator state." << std::endl;
        try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_before_gemmA_Bt_sgemm.json")); } catch(...) {}
    }
    cublasStatus_t status_f = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_cm, M, B_cm, K, &beta, C_cm, M);
    LOG_INFO_STREAM("cublasSgemm completed with status=" << (int)status_f << " (" << cublasStatusName(status_f) << ")");
    if(status_f == CUBLAS_STATUS_INTERNAL_ERROR) {
        std::cerr << "cublasSgemm returned INTERNAL_ERROR. Running device sanity check and serializing state." << std::endl;
        if(!performDeviceSanityCheck()) {
            try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_on_internal_error.json")); } catch(...) {}
        }
    }
    if(status_f == CUBLAS_STATUS_SUCCESS)
    {
        // Transpose C_cm back to row-major C (M x N)
        launchColToRowMajor(C_cm, C, M, N);
        // Free temp buffers (respecting pool vs tracked alloc)
        if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt cleanup");
        if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_Bt cleanup");
        if(C_fromPool) RuntimeMemory::releaseGPU(C_cm); else AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_Bt cleanup");
        std::cout << "cuBLAS gemmA_Bt: transpose fallback succeeded" << std::endl;
        return;
    }
    else
    {
        std::cerr << "cuBLAS gemmA_Bt transpose fallback failed: " << (int)status_f << " (" << cublasStatusName(status_f) << ")" << std::endl;
        // On failure, attempt a retry using tracked allocations (avoid pool) to rule out pool-related corruption
        std::cerr << "cuBLAS gemmA_Bt: retrying with tracked cudaMalloc allocations (avoid pool)..." << std::endl;
        // Release any pool allocations
        if(A_fromPool) { RuntimeMemory::releaseGPU(A_cm); A_cm = nullptr; A_fromPool = false; }
        if(B_fromPool) { RuntimeMemory::releaseGPU(B_cm); B_cm = nullptr; B_fromPool = false; }
        if(C_fromPool) { RuntimeMemory::releaseGPU(C_cm); C_cm = nullptr; C_fromPool = false; }
        // Allocate with trackedMalloc
        cudaError_t aerr = AllocationTracker::trackedCudaMalloc((void**)&A_cm, A_cm_bytes, "CUBLASWrapper::gemmA_Bt retry A_cm");
        cudaError_t berr = AllocationTracker::trackedCudaMalloc((void**)&B_cm, B_cm_bytes, "CUBLASWrapper::gemmA_Bt retry B_cm");
        cudaError_t cerr_ = AllocationTracker::trackedCudaMalloc((void**)&C_cm, C_cm_bytes, "CUBLASWrapper::gemmA_Bt retry C_cm");
        if(aerr != cudaSuccess || berr != cudaSuccess || cerr_ != cudaSuccess) {
            std::cerr << "Retry allocation failed: " << cudaGetErrorString(aerr) << ", " << cudaGetErrorString(berr) << ", " << cudaGetErrorString(cerr_) << std::endl;
            if(A_cm) AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt retry cleanup");
            if(B_cm) AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_Bt retry cleanup");
            if(C_cm) AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_Bt retry cleanup");
            cublasCheck(status_f);
        }
        // Recompute transposes into new buffers
        launchRowToColMajor(A, A_cm, M, K);
        launchRowToColMajorTranspose(B, B_cm, N, K);
        LOG_INFO_STREAM("Retry cublasSgemm with tracked buffers A_cm="<<(void*)A_cm<<" B_cm="<<(void*)B_cm<<" C_cm="<<(void*)C_cm);
        cublasStatus_t status2 = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_cm, M, B_cm, K, &beta, C_cm, M);
        LOG_INFO_STREAM("Retry cublasSgemm completed with status="<< (int)status2 << " ("<< cublasStatusName(status2) <<")");
        if(status2 == CUBLAS_STATUS_SUCCESS) {
            launchColToRowMajor(C_cm, C, M, N);
            AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt retry free A");
            AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_Bt retry free B");
            AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_Bt retry free C");
            std::cout<<"cuBLAS gemmA_Bt: retry succeeded with tracked allocations"<<std::endl;
            return;
        } else {
            std::cerr<<"Retry cublasSgemm failed: "<<(int)status2<<" ("<<cublasStatusName(status2)<<")"<<std::endl;
            AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_Bt retry fail free A");
            AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_Bt retry fail free B");
            AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_Bt retry fail free C");
            cublasCheck(status2);
        }
    }

    cublasCheck(status_f);
}

void CUBLASContext::gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    if(!initialized_) initialize();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if(stream != 0)
    {
        cublasCheck(cublasSetStream(handle_, stream));
    }

    // Prefer robust column-major transpose fallback to avoid illegal parameter errors in cublas mappings.
    std::cerr << "cuBLAS gemm: using transpose-conversion fallback (robust path)" << std::endl;

    float* A_cm = nullptr; // M x K col-major
    float* B_cm = nullptr; // K x N col-major
    float* C_cm = nullptr; // M x N col-major

    size_t A_cm_bytes = (size_t)M * K * sizeof(float);
    size_t B_cm_bytes = (size_t)K * N * sizeof(float);
    size_t C_cm_bytes = (size_t)M * N * sizeof(float);

    bool A_fromPool = false, B_fromPool = false, C_fromPool = false;
#ifndef FORCE_TRACKED_SCRATCH
    void* tmp = RuntimeMemory::allocateScratchGPU(A_cm_bytes);
#else
    void* tmp = nullptr;
#endif
    if(tmp) { A_cm = (float*)tmp; A_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&A_cm, A_cm_bytes, "CUBLASWrapper::gemm A_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc A_cm failed: " << cerr << std::endl; cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(B_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { B_cm = (float*)tmp; B_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&B_cm, B_cm_bytes, "CUBLASWrapper::gemm B_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc B_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm cleanup"); cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(C_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { C_cm = (float*)tmp; C_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&C_cm, C_cm_bytes, "CUBLASWrapper::gemm C_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc C_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm cleanup"); if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemm cleanup"); cublasCheck(CUBLAS_STATUS_ALLOC_FAILED); }
    }

    // A is row-major M x K -> A_cm column-major M x K
    launchRowToColMajor(A, A_cm, M, K);
    // B is row-major K x N? In our callers B often is vHead which is (totalSeq x headDim) row-major; we need B_cm as K x N
    // For general row-major B (rows = N, cols = K), use transpose kernel to produce K x N col-major
    launchRowToColMajorTranspose(B, B_cm, N, K);

    LOG_INFO_STREAM("cublasSgemm about to run (gemm). M=" << M << " N=" << N << " K=" << K << " A_cm=" << (void*)A_cm << " B_cm=" << (void*)B_cm << " C_cm=" << (void*)C_cm << " A_fromPool=" << A_fromPool << " B_fromPool=" << B_fromPool << " C_fromPool=" << C_fromPool);
    if(!performDeviceSanityCheck()) {
        std::cerr << "Device sanity check FAILED before cublasSgemm (gemm). Serializing allocator state." << std::endl;
        try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_before_gemm_sgemm.json")); } catch(...) {}
    }
    cublasStatus_t status_f = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_cm, M, B_cm, K, &beta, C_cm, M);
    LOG_INFO_STREAM("cublasSgemm (gemm) completed with status=" << (int)status_f << " (" << cublasStatusName(status_f) << ")");
    if(status_f == CUBLAS_STATUS_INTERNAL_ERROR) {
        std::cerr << "cublasSgemm returned INTERNAL_ERROR (gemm). Running device sanity check and serializing state." << std::endl;
        if(!performDeviceSanityCheck()) {
            try { RuntimeMemory::serializePoolState(std::string("out/allocator_state_on_internal_error_gemm.json")); } catch(...) {}
        }
    }
    if(status_f == CUBLAS_STATUS_SUCCESS)
    {
        // transpose back
        launchColToRowMajor(C_cm, C, M, N);
        // Free temp buffers (respect pool vs tracked)
        if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm cleanup");
        if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemm cleanup");
        if(C_fromPool) RuntimeMemory::releaseGPU(C_cm); else AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemm cleanup");
        std::cout << "cuBLAS gemm transpose fallback succeeded" << std::endl;
        return;
    }

    std::cerr << "cuBLAS gemm transpose fallback failed: " << (int)status_f << " (" << cublasStatusName(status_f) << ")" << std::endl;
    // Retry with tracked cudaMalloc allocations to rule out pool corruption
    std::cerr << "cuBLAS gemm: retrying with tracked cudaMalloc allocations (avoid pool)..." << std::endl;
    if(A_fromPool) { RuntimeMemory::releaseGPU(A_cm); A_cm = nullptr; A_fromPool = false; }
    if(B_fromPool) { RuntimeMemory::releaseGPU(B_cm); B_cm = nullptr; B_fromPool = false; }
    if(C_fromPool) { RuntimeMemory::releaseGPU(C_cm); C_cm = nullptr; C_fromPool = false; }

    cudaError_t aerr = AllocationTracker::trackedCudaMalloc((void**)&A_cm, A_cm_bytes, "CUBLASWrapper::gemm retry A_cm");
    cudaError_t berr = AllocationTracker::trackedCudaMalloc((void**)&B_cm, B_cm_bytes, "CUBLASWrapper::gemm retry B_cm");
    cudaError_t cerr_ = AllocationTracker::trackedCudaMalloc((void**)&C_cm, C_cm_bytes, "CUBLASWrapper::gemm retry C_cm");
    if(aerr != cudaSuccess || berr != cudaSuccess || cerr_ != cudaSuccess) {
        std::cerr << "Retry allocation failed: " << cudaGetErrorString(aerr) << ", " << cudaGetErrorString(berr) << ", " << cudaGetErrorString(cerr_) << std::endl;
        if(A_cm) AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm retry cleanup");
        if(B_cm) AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemm retry cleanup");
        if(C_cm) AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemm retry cleanup");
        cublasCheck(status_f);
    }
    // Recompute transposes into new buffers
    launchRowToColMajor(A, A_cm, M, K);
    launchRowToColMajorTranspose(B, B_cm, N, K);
    LOG_INFO_STREAM("Retry cublasSgemm with tracked buffers A_cm="<<(void*)A_cm<<" B_cm="<<(void*)B_cm<<" C_cm="<<(void*)C_cm);
    cublasStatus_t status2 = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A_cm, M, B_cm, K, &beta, C_cm, M);
    LOG_INFO_STREAM("Retry cublasSgemm completed with status=" << (int)status2 << " (" << cublasStatusName(status2) << ")");
    if(status2 == CUBLAS_STATUS_SUCCESS)
    {
        launchColToRowMajor(C_cm, C, M, N);
        AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm retry free A");
        AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemm retry free B");
        AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemm retry free C");
        std::cout << "cuBLAS gemm: retry succeeded with tracked allocations" << std::endl;
        return;
    }

    std::cerr << "Retry cublasSgemm failed: " << (int)status2 << " (" << cublasStatusName(status2) << ")" << std::endl;
    AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemm retry fail free A");
    AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemm retry fail free B");
    AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemm retry fail free C");

    cublasCheck(status_f);
}

void CUBLASContext::gemmA_BtStridedBatched(const float* A, const float* B, float* C, int M, int N, int K,
                                           long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream)
{
    if(!initialized_) initialize();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if(stream != 0) cublasCheck(cublasSetStream(handle_, stream));

    // Validate input pointers
    validateDevicePointer(A, "A_strided");
    validateDevicePointer(B, "B_strided");
    validateDevicePointer(C, "C_strided");

    // Try strided-batched using column-major conversion to avoid row-major mapping issues.
    // Allocate temporary column-major batched buffers and convert all matrices to column-major in-place.
    size_t A_cm_bytes = (size_t)batchCount * M * K * sizeof(float);
    size_t B_cm_bytes = (size_t)batchCount * K * N * sizeof(float);
    size_t C_cm_bytes = (size_t)batchCount * M * N * sizeof(float);

    float* A_cm = nullptr; float* B_cm = nullptr; float* C_cm = nullptr;
    bool A_fromPool = false, B_fromPool = false, C_fromPool = false;
#ifndef FORCE_TRACKED_SCRATCH
    void* tmp = RuntimeMemory::allocateScratchGPU(A_cm_bytes);
#else
    void* tmp = nullptr;
#endif
    if(tmp) { A_cm = (float*)tmp; A_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&A_cm, A_cm_bytes, "CUBLASWrapper::gemmA_BtStridedBatched A_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc A_cm failed: " << cerr << std::endl; goto sb_fallback; }
    }
#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(B_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { B_cm = (float*)tmp; B_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&B_cm, B_cm_bytes, "CUBLASWrapper::gemmA_BtStridedBatched B_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc B_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::sb cleanup"); goto sb_fallback; }
    }
#ifndef FORCE_TRACKED_SCRATCH
    tmp = RuntimeMemory::allocateScratchGPU(C_cm_bytes);
#else
    tmp = nullptr;
#endif
    if(tmp) { C_cm = (float*)tmp; C_fromPool = true; }
    else {
        cudaError_t cerr = AllocationTracker::trackedCudaMalloc((void**)&C_cm, C_cm_bytes, "CUBLASWrapper::gemmA_BtStridedBatched C_cm");
        if(cerr != cudaSuccess) { std::cerr << "cudaMalloc C_cm failed: " << cerr << std::endl; if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::sb cleanup"); if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::sb cleanup"); goto sb_fallback; }
    }

    // Convert each batch element to column-major contiguous region
    for(int i=0;i<batchCount;i++)
    {
        const float* Ai = A + (size_t)strideA * i;
        const float* Bi = B + (size_t)strideB * i;
        float* Aci = A_cm + (size_t)i * (M * K);
        float* Bci = B_cm + (size_t)i * (K * N);
        float* Cci = C_cm + (size_t)i * (M * N);

        launchRowToColMajor(Ai, Aci, M, K);
        // B is row-major N x K (rows=N cols=K) -> produce K x N col-major
        launchRowToColMajorTranspose(Bi, Bci, N, K);
    }

    // Now call cublasSgemmStridedBatched on column-major contiguous arrays
    cublasStatus_t status;
    status = cublasSgemmStridedBatched(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        /*m=*/M,
        /*n=*/N,
        /*k=*/K,
        &alpha,
        A_cm,
        /*lda=*/M,
        /*strideA=*/(long long)(M * K),
        B_cm,
        /*ldb=*/K,
        /*strideB=*/(long long)(K * N),
        &beta,
        C_cm,
        /*ldc=*/M,
        /*strideC=*/(long long)(M * N),
        batchCount
    );

    if(status == CUBLAS_STATUS_SUCCESS)
    {
        // Convert C_cm back to row-major per-batch
        for(int i=0;i<batchCount;i++)
        {
            float* Cci = C_cm + (size_t)i * (M * N);
            float* Ci = C + (size_t)strideC * i;
            launchColToRowMajor(Cci, Ci, M, N);
        }
        // Free temp buffers (respecting pool vs tracked alloc)
        if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");
        if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");
        if(C_fromPool) RuntimeMemory::releaseGPU(C_cm); else AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");
        return;
    }

    std::cerr << "cublasSgemmStridedBatched column-major path failed: " << (int)status << " (" << cublasStatusName(status) << ")\n";
    if(A_fromPool) RuntimeMemory::releaseGPU(A_cm); else AllocationTracker::trackedCudaFree(A_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");
    if(B_fromPool) RuntimeMemory::releaseGPU(B_cm); else AllocationTracker::trackedCudaFree(B_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");
    if(C_fromPool) RuntimeMemory::releaseGPU(C_cm); else AllocationTracker::trackedCudaFree(C_cm, "CUBLASWrapper::gemmA_BtStridedBatched cleanup");

sb_fallback:
    std::cerr << "Falling back to per-matrix gemmA_Bt\n";
    for(int i=0;i<batchCount;i++)
    {
        const float* Ai = A + (size_t)strideA * i;
        const float* Bi = B + (size_t)strideB * i;
        float* Ci = C + (size_t)strideC * i;
        try { gemmA_Bt(Ai, Bi, Ci, M, N, K, stream); }
        catch (...) { /* continue fallback */ }
    }
}

void CUBLASContext::gemmStridedBatched(const float* A, const float* B, float* C, int M, int N, int K,
                                        long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream)
{
    if(!initialized_) initialize();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if(stream != 0) cublasCheck(cublasSetStream(handle_, stream));

    // Validate pointers and dimensions before calling cuBLAS
    validateDevicePointer(A, "A_sb");
    validateDevicePointer(B, "B_sb");
    validateDevicePointer(C, "C_sb");
    if(M <= 0 || N <= 0 || K <= 0 || batchCount <= 0)
    {
        std::cerr << "Invalid GEMM dimensions: M=" << M << " N=" << N << " K=" << K << " batch=" << batchCount << std::endl;
        throw std::runtime_error("Invalid GEMM dimensions");
    }

    // First attempt: column-major contiguous batched GEMM (A_cm: MxK, B_cm: KxN, C_cm: MxN, column-major)
    cublasStatus_t status = cublasSgemmStridedBatched(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        /*m=*/M,
        /*n=*/N,
        /*k=*/K,
        &alpha,
        A,
        /*lda=*/M,
        /*strideA=*/strideA,
        B,
        /*ldb=*/K,
        /*strideB=*/strideB,
        &beta,
        C,
        /*ldc=*/M,
        /*strideC=*/strideC,
        batchCount
    );

    if(status == CUBLAS_STATUS_SUCCESS)
    {
        return;
    }

    std::cerr << "cublasSgemmStridedBatched (gemm) failed (column-major attempt): " << (int)status << " (" << cublasStatusName(status) << ") - falling back to legacy row-major mapping\n";

    // Legacy row-major attempt (existing callers may pass row-major buffers)
    status = cublasSgemmStridedBatched(
        handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        /*m=*/M,
        /*n=*/N,
        /*k=*/K,
        &alpha,
        A,
        /*lda=*/K,
        /*strideA=*/strideA,
        B,
        /*ldb=*/N,
        /*strideB=*/strideB,
        &beta,
        C,
        /*ldc=*/N,
        /*strideC=*/strideC,
        batchCount
    );

    if(status == CUBLAS_STATUS_SUCCESS)
    {
        return;
    }

    std::cerr << "cublasSgemmStridedBatched (gemm) failed (row-major attempt): " << (int)status << " (" << cublasStatusName(status) << ") - falling back to per-matrix gemm\n";

    for(int i=0;i<batchCount;i++)
    {
        const float* Ai = A + (size_t)strideA * i;
        const float* Bi = B + (size_t)strideB * i;
        float* Ci = C + (size_t)strideC * i;
        try { gemm(Ai, Bi, Ci, M, N, K, stream); } catch(...){}
    }
}

void CUBLASContext::gemmA_BtStridedBatchedCM(const float* A_cm, const float* B_cm, float* C_cm, int M, int N, int K,
                                   long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream)
{
    if(!initialized_) initialize();
    if(stream != 0) cublasCheck(cublasSetStream(handle_, stream));
    const float alpha = 1.0f; const float beta = 0.0f;

    // Column-major: compute C_cm = A_cm (M x K) * B_cm (K x N)
    cublasStatus_t status = cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        &alpha,
        A_cm, /*lda=*/M, /*strideA=*/strideA,
        B_cm, /*ldb=*/K, /*strideB=*/strideB,
        &beta,
        C_cm, /*ldc=*/M, /*strideC=*/strideC,
        batchCount
    );

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasSgemmStridedBatched (CM) failed: " << (int)status << " (" << cublasStatusName(status) << ") - falling back to per-matrix gemmA_Bt\n";
        for(int i=0;i<batchCount;i++)
        {
            const float* Ai = A_cm + (size_t)strideA * i;
            const float* Bi = B_cm + (size_t)strideB * i;
            float* Ci = C_cm + (size_t)strideC * i;
            try { gemmA_Bt(Ai, Bi, Ci, M, N, K, stream); } catch(...){}
        }
    }
}

void CUBLASContext::gemmStridedBatchedCM(const float* A_cm, const float* B_cm, float* C_cm, int M, int N, int K,
                               long long strideA, long long strideB, long long strideC, int batchCount, cudaStream_t stream)
{
    if(!initialized_) initialize();
    if(stream != 0) cublasCheck(cublasSetStream(handle_, stream));
    const float alpha = 1.0f; const float beta = 0.0f;

    // Column-major batched GEMM
    cublasStatus_t status = cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        &alpha,
        A_cm, /*lda=*/M, /*strideA=*/strideA,
        B_cm, /*ldb=*/K, /*strideB=*/strideB,
        &beta,
        C_cm, /*ldc=*/M, /*strideC=*/strideC,
        batchCount
    );

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasSgemmStridedBatched (CM gemm) failed: " << (int)status << " (" << cublasStatusName(status) << ") - falling back to per-matrix gemm\n";
        for(int i=0;i<batchCount;i++)
        {
            const float* Ai = A_cm + (size_t)strideA * i;
            const float* Bi = B_cm + (size_t)strideB * i;
            float* Ci = C_cm + (size_t)strideC * i;
            try { gemm(Ai, Bi, Ci, M, N, K, stream); } catch(...){}
        }
    }
}
