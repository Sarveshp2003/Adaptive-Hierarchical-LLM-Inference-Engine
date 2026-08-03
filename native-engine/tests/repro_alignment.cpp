#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    size_t N = 1024 * 1024; // 1MB
    void* host = nullptr;
    cudaError_t err = cudaHostAlloc(&host, N, cudaHostAllocDefault);
    std::cout << "cudaHostAlloc -> " << host << " err=" << cudaGetErrorString(err) << "\n";

    void* dev = nullptr;
    err = cudaMalloc(&dev, N);
    std::cout << "cudaMalloc -> " << dev << " err=" << cudaGetErrorString(err) << "\n";

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // Normal async copy
    err = cudaMemcpyAsync(dev, host, N, cudaMemcpyHostToDevice, stream);
    std::cout << "cudaMemcpyAsync(dev,host) -> " << cudaGetErrorString(err) << "\n";
    cudaStreamSynchronize(stream);
    std::cout << "cudaStreamSynchronize() done\n";

    // Misaligned host pointer (offset by 1)
    void* host_mis = reinterpret_cast<char*>(host) + 1;
    err = cudaMemcpyAsync(dev, host_mis, N-1, cudaMemcpyHostToDevice, stream);
    std::cout << "cudaMemcpyAsync(dev,host+1) -> " << cudaGetErrorString(err) << "\n";
    cudaStreamSynchronize(stream);

    // Misaligned device pointer (offset by 1)
    void* dev_mis = reinterpret_cast<char*>(dev) + 1;
    err = cudaMemcpyAsync(dev_mis, host, N-1, cudaMemcpyHostToDevice, stream);
    std::cout << "cudaMemcpyAsync(dev+1,host) -> " << cudaGetErrorString(err) << "\n";
    cudaStreamSynchronize(stream);

    // Try a small GEMM using cuBLAS if available (best-effort): allocate tiny matrices
    // Note: linking against cublas requires adaptive_engine; this repro focuses on memcpy/alignment.

    cudaFree(dev);
    cudaFreeHost(host);
    cudaStreamDestroy(stream);

    std::cout << "Repro finished\n";
    return 0;
}
