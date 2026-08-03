#include "MultiHeadAttention.h"
#include "Linear.h"
#include "CUBLASWrapper.h"
#include "CUDAError.h"
#include "CUDAStream.h"
#include "Logger.h"
#include "RuntimeMemory.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

// CUDA kernel launchers (defined in cuda/*.cu)
extern "C" void launchMatmulTranspose(float* A, float* B, float* C, int M, int N, int K);
extern "C" void launchMatmul(float* A, float* B, float* C, int M, int K, int N);
extern "C" void launchScaledSoftmax(float* input, float* output, int rows, int cols, float scale);
extern "C" void launchScaledSoftmaxBatched(float* data, int batchCount, int rows, int cols, float scale);
extern "C" void launchGatherHead(const float* src, float* dst, int rows, int headDim, int headOffset, int stride);
extern "C" void launchRowToColMajor(const float* src, float* dst, int rows, int cols);
extern "C" void launchRowToColMajorTranspose(const float* src, float* dst, int N, int K);
extern "C" void launchColToRowMajor(const float* src, float* dst, int rows, int cols);


MultiHeadAttention::MultiHeadAttention(
    int hiddenSize,
    int numHeads
)
:
hiddenSize(hiddenSize),
numHeads(numHeads),
headDim(hiddenSize / numHeads),
qLinear(hiddenSize, hiddenSize),
kLinear(hiddenSize, hiddenSize),
vLinear(hiddenSize, hiddenSize),
outLinear(hiddenSize, hiddenSize)

{

    if(hiddenSize % numHeads != 0)
    {
        throw std::runtime_error(
            "hiddenSize must be divisible by numHeads"
        );
    }


}


void MultiHeadAttention::forward(
    Tensor& input,
    Tensor& output,
    KVCache* cache,
    bool useCache
)
{
    // For small or awkward shapes, prefer the CPU reference path to keep the runtime stable.
    if(input.rank() != 3 || input.shape()[0] <= 1 || input.shape()[1] <= 4 || input.elements() <= 64)
    {
        forwardCPU(input, output, cache, useCache);
        return;
    }

    // GPU-only attention: project Q/K/V on GPU and compute attention fully on device.
    
    // Validate input layout: expect [batch, seq, hidden]
    if(input.shape().size() != 3)
    {
        throw std::runtime_error("MultiHeadAttention::forward expected input rank 3 [batch,seq,hidden]");
    }

    int batch = input.shape()[0];
    int seqLen = input.shape()[1];
    int inHidden = input.shape()[2];

    LOG_INFO_STREAM("MultiHeadAttention::forward batch=" << batch << " seqLen=" << seqLen << " hidden=" << inHidden << " numHeads=" << numHeads << " headDim=" << headDim);

    if(inHidden != hiddenSize)
    {
        throw std::runtime_error("MultiHeadAttention::forward input hidden dimension does not match layer hidden size");
    }

    // Profiling timers (milliseconds)
    float t_qkv = 0.0f, t_qk = 0.0f, t_softmax = 0.0f, t_av = 0.0f, t_out = 0.0f;

    // temporary timing helpers
    float t_seg = 0.0f;
    float t_seg2 = 0.0f;

    // column-major temporary pointers (allocated per-batch)
    float* qHeads_cm = nullptr;
    float* kHeads_cm = nullptr;
    float* vHeads_cm = nullptr;
    float* scoresAll_cm = nullptr;
    float* headContextsAll_cm = nullptr;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    cudaStream_t stream = CUDAStream::get();

    Tensor Q(input.shape(), DataType::FP32);
    Tensor K(input.shape(), DataType::FP32);
    Tensor V(input.shape(), DataType::FP32);

    Q.allocateCPU(); Q.allocateGPU();
    K.allocateCPU(); K.allocateGPU();
    V.allocateCPU(); V.allocateGPU();

    // Linear projections write directly to GPU memory
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    qLinear.forward(input, Q);
    kLinear.forward(input, K);
    vLinear.forward(input, V);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_qkv, ev_start, ev_stop));

    // Use cache only for batch==1 and when requested
    int cacheLen = 0;
    if(cache && useCache)
    {
        cacheLen = cache->seqLen;
        if(batch != 1 && cacheLen > 0)
        {
            cacheLen = 0; // ignore cache for now if batch!=1
        }
    }

    int totalSeq = seqLen + cacheLen;
    float scale = 1.0f / sqrtf((float)headDim);

    // Prepare device-side total K/V if cache present
    // Use standardized [batch, seq, hidden] layout; when using cache, batch==1 and first dim is 1
    Tensor kTotal({1, totalSeq, hiddenSize}, DataType::FP32);
    Tensor vTotal({1, totalSeq, hiddenSize}, DataType::FP32);
    bool usingTotalKV = false;

    if(cacheLen > 0)
    {
        usingTotalKV = true;

        // allocate CPU buffer to stage cached entries, upload, then copy device K into tail
        kTotal.allocateCPU(); kTotal.allocateGPU();
        vTotal.allocateCPU(); vTotal.allocateGPU();

        // zero-initialize
        memset(kTotal.cpu(), 0, kTotal.bytes());
        memset(vTotal.cpu(), 0, vTotal.bytes());

        if(!cache->keys.empty())
            memcpy(kTotal.cpu(), cache->keys.data(), sizeof(float) * cache->keys.size());
        if(!cache->vals.empty())
            memcpy(vTotal.cpu(), cache->vals.data(), sizeof(float) * cache->vals.size());

        kTotal.upload();
        vTotal.upload();

        // copy current K/V device data into the tail region (device->device)
        size_t tailBytes = (size_t)seqLen * (size_t)hiddenSize * sizeof(float);
        float* kDest = kTotal.gpu() + static_cast<size_t>(kTotal.offset(0, cacheLen, 0));
        float* vDest = vTotal.gpu() + static_cast<size_t>(vTotal.offset(0, cacheLen, 0));

        CUDA_CHECK(cudaMemcpy(kDest, K.gpu(), tailBytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(vDest, V.gpu(), tailBytes, cudaMemcpyDeviceToDevice));
    }

    // Prepare output context on GPU
    Tensor context(input.shape(), DataType::FP32);
    context.allocateGPU();
    // zero context on device
    CUDA_CHECK(cudaMemset(context.gpu(), 0, context.bytes()));

    // Per-head attention on GPU (batched across heads per batch item)
    for(int b=0;b<batch;b++)
    {
        int kRows = usingTotalKV ? totalSeq : seqLen;

        // Allocate batched head buffers: [numHeads, rows, headDim]
        Tensor qHeads({numHeads, seqLen, headDim}, DataType::FP32);
        Tensor kHeads({numHeads, kRows, headDim}, DataType::FP32);
        Tensor vHeads({numHeads, kRows, headDim}, DataType::FP32);
        Tensor scoresAll({numHeads, seqLen, kRows}, DataType::FP32);
        Tensor headContextsAll({numHeads, seqLen, headDim}, DataType::FP32);

        qHeads.allocateGPU();
        kHeads.allocateGPU();
        vHeads.allocateGPU();
        scoresAll.allocateGPU();
        headContextsAll.allocateGPU();

        // Gather per-head matrices into the batched buffers
        const float* qSrcBase = Q.gpu() + static_cast<size_t>(Q.offset(b, 0, 0));
        const float* kSrcBase = usingTotalKV ? (kTotal.gpu() + static_cast<size_t>(kTotal.offset(0, 0, 0))) : (K.gpu() + static_cast<size_t>(K.offset(b, 0, 0)));
        const float* vSrcBase = usingTotalKV ? (vTotal.gpu() + static_cast<size_t>(vTotal.offset(0, 0, 0))) : (V.gpu() + static_cast<size_t>(V.offset(b, 0, 0)));

        for(int h=0; h<numHeads; ++h)
        {
            int headOffset = h * headDim;
            float* qDst = qHeads.gpu() + static_cast<size_t>(qHeads.offset(h, 0, 0));
            float* kDst = kHeads.gpu() + static_cast<size_t>(kHeads.offset(h, 0, 0));
            float* vDst = vHeads.gpu() + static_cast<size_t>(vHeads.offset(h, 0, 0));

            launchGatherHead(qSrcBase, qDst, seqLen, headDim, headOffset, hiddenSize);
            launchGatherHead(kSrcBase, kDst, kRows, headDim, headOffset, hiddenSize);
            launchGatherHead(vSrcBase, vDst, kRows, headDim, headOffset, hiddenSize);
        }
        // Check for kernel launch errors from gather
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Attempt column-major path: allocate CM buffers
        size_t q_cm_elems = (size_t)numHeads * seqLen * headDim;
        size_t k_cm_elems = (size_t)numHeads * headDim * kRows; // will store K x N per-head as K=headDim, N=kRows
        size_t v_cm_elems = (size_t)numHeads * headDim * kRows;

        float* qHeads_cm_local = static_cast<float*>(RuntimeMemory::allocateGPU(q_cm_elems * sizeof(float)));
        float* kHeads_cm_local = static_cast<float*>(RuntimeMemory::allocateGPU(k_cm_elems * sizeof(float)));
        float* vHeads_cm_local = static_cast<float*>(RuntimeMemory::allocateGPU(v_cm_elems * sizeof(float)));

        if(qHeads_cm_local && kHeads_cm_local && vHeads_cm_local)
        {
            // For each head, convert row-major (rows=seqLen, cols=headDim) to column-major contiguous
            for(int h=0; h<numHeads; ++h)
            {
                float* qSrc = qHeads.gpu() + static_cast<size_t>(qHeads.offset(h,0,0));
                float* qDst = qHeads_cm_local + (size_t)h * (seqLen * headDim);
                launchRowToColMajor(qSrc, qDst, seqLen, headDim);

                float* kSrc = kHeads.gpu() + static_cast<size_t>(kHeads.offset(h,0,0));
                float* kDst = kHeads_cm_local + (size_t)h * (headDim * kRows);
                // convert row-major N x K to K x N col-major
                launchRowToColMajorTranspose(kSrc, kDst, kRows, headDim);

                float* vSrc = vHeads.gpu() + static_cast<size_t>(vHeads.offset(h,0,0));
                float* vDst = vHeads_cm_local + (size_t)h * (headDim * kRows);
                launchRowToColMajorTranspose(vSrc, vDst, kRows, headDim);
            }
            // check for conversion kernel errors
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Prepare column-major scores and headContext buffers
            size_t scores_cm_elems = (size_t)numHeads * seqLen * kRows; // each C is M x N (seqLen x kRows)
            float* scoresAll_cm_local = static_cast<float*>(RuntimeMemory::allocateGPU(scores_cm_elems * sizeof(float)));
            float* headContextsAll_cm_local = static_cast<float*>(RuntimeMemory::allocateGPU((size_t)numHeads * seqLen * headDim * sizeof(float)));

            if(scoresAll_cm_local && headContextsAll_cm_local)
            {
                // Compute Q * K^T for all heads in a single column-major strided-batched GEMM
                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                try {
                    long long strideA = (long long)seqLen * headDim; // in elements per A_cm (M*K)
                    long long strideB = (long long)headDim * kRows;  // in elements per B_cm (K*N)
                    long long strideC = (long long)seqLen * kRows;   // in elements per C_cm (M*N)

                    CUBLASContext::instance().gemmA_BtStridedBatchedCM(
                        qHeads_cm_local,
                        kHeads_cm_local,
                        scoresAll_cm_local,
                        seqLen,
                        kRows,
                        headDim,
                        strideA,
                        strideB,
                        strideC,
                        numHeads,
                        stream
                    );
                } catch (const std::exception &e) {
                    std::cerr << "strided-batched column-major cuBLAS gemmA_Bt failed: " << e.what() << " - falling back to legacy path\n";
                }
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaEventElapsedTime(&t_seg2, ev_start, ev_stop)); t_qk += t_seg2;

                LOG_INFO_STREAM("Completed batched QK matmul (column-major) for batch " << b << " (numHeads=" << numHeads << ")");

                // Convert scoresAll_cm (column-major MxN per head) into row-major scoresAll Tensor for softmax
                for(int h=0; h<numHeads; ++h)
                {
                    float* srcCm = scoresAll_cm_local + (size_t)h * (seqLen * kRows);
                    float* dstRow = scoresAll.gpu() + static_cast<size_t>(scoresAll.offset(h,0,0));
                    launchColToRowMajor(srcCm, dstRow, seqLen, kRows);
                }

                // Batched softmax across heads (operates on row-major scoresAll)
                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                launchScaledSoftmaxBatched(scoresAll.gpu(), numHeads, seqLen, kRows, scale);
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventElapsedTime(&t_seg2, ev_start, ev_stop)); t_softmax += t_seg2;

                // Now convert scores (row-major) back to column-major for GEMM with V_cm
                for(int h=0; h<numHeads; ++h)
                {
                    float* dstCm = scoresAll_cm_local + (size_t)h * (seqLen * kRows);
                    float* srcRow = scoresAll.gpu() + static_cast<size_t>(scoresAll.offset(h,0,0));
                    // srcRow is row-major M x N; convert to column-major in dstCm
                    launchRowToColMajor(srcRow, dstCm, seqLen, kRows);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                // Compute scores * V -> headContexts for all heads using column-major GEMM
                CUDA_CHECK(cudaEventRecord(ev_start, stream));
                try {
                    long long strideA2 = (long long)seqLen * kRows;    // scoresAll_cm per-head size (M*N)
                    long long strideB2 = (long long)headDim * kRows;   // vHeads_cm per-head size (K*N) here K=headDim, N=kRows
                    long long strideC2 = (long long)seqLen * headDim;  // headContext per-head size (M*K)

                    CUBLASContext::instance().gemmStridedBatchedCM(
                        scoresAll_cm_local,
                        vHeads_cm_local,
                        headContextsAll_cm_local,
                        seqLen,
                        headDim,
                        kRows,
                        strideA2,
                        strideB2,
                        strideC2,
                        numHeads,
                        stream
                    );
                } catch (const std::exception &e) {
                    std::cerr << "strided-batched column-major cuBLAS gemm (scores*V) failed: " << e.what() << " - falling back to legacy path\n";
                }
                CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaEventElapsedTime(&t_seg2, ev_start, ev_stop)); t_av += t_seg2;

                LOG_INFO_STREAM("Completed batched AV matmul (column-major) for batch " << b << " (numHeads=" << numHeads << ")");

                // Convert headContextsAll_cm (column-major M x K per head) into row-major headContextsAll Tensor
                for(int h=0; h<numHeads; ++h)
                {
                    float* srcCm = headContextsAll_cm_local + (size_t)h * (seqLen * headDim);
                    float* dstRow = headContextsAll.gpu() + static_cast<size_t>(headContextsAll.offset(h,0,0));
                    launchColToRowMajor(srcCm, dstRow, seqLen, headDim);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                // release column-major temps
                RuntimeMemory::releaseGPU(qHeads_cm_local); RuntimeMemory::releaseGPU(kHeads_cm_local); RuntimeMemory::releaseGPU(vHeads_cm_local);
                RuntimeMemory::releaseGPU(scoresAll_cm_local); RuntimeMemory::releaseGPU(headContextsAll_cm_local);

            }

                // proceed to post-batched handling
            else
            {
                // release partial
                RuntimeMemory::releaseGPU(scoresAll_cm_local);
                RuntimeMemory::releaseGPU(headContextsAll_cm_local);
                RuntimeMemory::releaseGPU(qHeads_cm_local); RuntimeMemory::releaseGPU(kHeads_cm_local); RuntimeMemory::releaseGPU(vHeads_cm_local);
                // fall through to legacy

                // Legacy per-head compute
                LOG_INFO_STREAM("Column-major path allocation failed; falling back to legacy per-head path");
                // fall through
            }
        }
        else
        {
            // Legacy per-head compute
            LOG_INFO_STREAM("Column-major path allocation failed; falling back to legacy per-head path");
        }

        // Legacy per-head code
        LOG_INFO_STREAM("Completed batched QK matmul for batch " << b << " (numHeads=" << numHeads << ")");

        // Batched softmax across heads
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        launchScaledSoftmaxBatched(scoresAll.gpu(), numHeads, seqLen, kRows, scale);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&t_seg, ev_start, ev_stop)); t_softmax += t_seg;

        // Compute scores * V -> headContexts for all heads in a single batched GEMM
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        try {
            long long strideA2 = (long long)seqLen * kRows;    // scores per-head size
            long long strideB2 = (long long)kRows * headDim;   // vHeads per-head size
            long long strideC2 = (long long)seqLen * headDim;  // headContext per-head size

            CUBLASContext::instance().gemmStridedBatched(
                scoresAll.gpu(),
                vHeads.gpu(),
                headContextsAll.gpu(),
                seqLen,
                headDim,
                kRows,
                strideA2,
                strideB2,
                strideC2,
                numHeads,
                stream
            );
        } catch (const std::exception &e) {
            std::cerr << "strided-batched cuBLAS gemm (scores*V) failed: " << e.what() << " - falling back to per-head matmuls\n";
            for(int h=0; h<numHeads; ++h)
            {
                float* sPtr = scoresAll.gpu() + static_cast<size_t>(scoresAll.offset(h,0,0));
                float* vPtr = vHeads.gpu() + static_cast<size_t>(vHeads.offset(h,0,0));
                float* outPtr = headContextsAll.gpu() + static_cast<size_t>(headContextsAll.offset(h,0,0));
                try {
                    CUBLASContext::instance().gemm(sPtr, vPtr, outPtr, seqLen, headDim, kRows, stream);
                } catch(...) {
                    launchMatmul(sPtr, vPtr, outPtr, seqLen, kRows, headDim);
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&t_seg, ev_start, ev_stop)); t_av += t_seg;

        LOG_INFO_STREAM("Completed batched AV matmul for batch " << b << " (numHeads=" << numHeads << ")");

        // Scatter each headContext into context tensor
        for(int h=0; h<numHeads; ++h)
        {
            int headOffset = h * headDim;
            float* ctxDest = context.gpu() + static_cast<size_t>(context.offset(b, 0, headOffset));
            float* srcPtr = headContextsAll.gpu() + static_cast<size_t>(headContextsAll.offset(h,0,0));

            size_t dstPitch = static_cast<size_t>(hiddenSize) * sizeof(float);
            size_t srcPitch = static_cast<size_t>(headDim) * sizeof(float);
            size_t widthBytes = static_cast<size_t>(headDim) * sizeof(float);
            size_t height = static_cast<size_t>(seqLen);

            CUDA_CHECK(cudaMemcpy2D(ctxDest, dstPitch, srcPtr, srcPitch, widthBytes, height, cudaMemcpyDeviceToDevice));
        }

        LOG_INFO_STREAM("Scattered heads into context for batch " << b);

        // release temporaries
        headContextsAll.release();
        scoresAll.release();
        qHeads.release();
        kHeads.release();
        vHeads.release();
    }

    // Output projection
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    outLinear.forward(context, output);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_out, ev_start, ev_stop));

    LOG_INFO_STREAM("Output projection done");

    // Output projection
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    outLinear.forward(context, output);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_out, ev_start, ev_stop));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Print profiling summary
    LOG_INFO_STREAM("MHA Profile (ms):");
    LOG_INFO_STREAM("  QKV Projection: " << t_qkv << " ms");
    LOG_INFO_STREAM("  QK Matmul:      " << t_qk << " ms");
    LOG_INFO_STREAM("  Softmax:        " << t_softmax << " ms");
    LOG_INFO_STREAM("  AV Matmul:      " << t_av << " ms");
    LOG_INFO_STREAM("  Output Project: " << t_out << " ms");

    // Update cache if requested and supported (batch==1)
    if(cache && useCache && cacheLen >= 0)
    {
        // Download current K/V tail back to CPU cache storage (only the newly added portion)
        // For now keep existing behavior: append current K/V from device to CPU cache
        std::vector<float> newK((size_t)seqLen * hiddenSize);
        std::vector<float> newV((size_t)seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(newK.data(), K.gpu(), (size_t)seqLen * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(newV.data(), V.gpu(), (size_t)seqLen * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));

        if(cacheLen == 0)
        {
            cache->keys = std::move(newK);
            cache->vals = std::move(newV);
        }
        else
        {
            int old = cache->seqLen;
            cache->keys.resize((old + seqLen) * hiddenSize);
            cache->vals.resize((old + seqLen) * hiddenSize);
            memcpy(cache->keys.data() + old * hiddenSize, newK.data(), sizeof(float) * newK.size());
            memcpy(cache->vals.data() + old * hiddenSize, newV.data(), sizeof(float) * newV.size());
        }
        cache->seqLen += seqLen;
    }

}
