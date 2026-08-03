#include "RuntimeMemory.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    RuntimeMemory::initializeGPU(64ULL * 1024 * 1024);

    constexpr size_t kBytes = 4ULL * 1024 * 1024;
    void* a = RuntimeMemory::allocateGPU(kBytes);
    void* b = RuntimeMemory::allocateGPU(kBytes);
    if (!a || !b) {
        std::cerr << "gpu allocation failed" << std::endl;
        RuntimeMemory::shutdown();
        return 2;
    }

    std::vector<unsigned char> payload(kBytes, 0x5A);
    cudaError_t copyErr = cudaMemcpy(b, payload.data(), kBytes, cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyErr) << std::endl;
        RuntimeMemory::releaseGPU(a);
        RuntimeMemory::releaseGPU(b);
        RuntimeMemory::shutdown();
        return 3;
    }

    cudaDeviceSynchronize();
    RuntimeMemory::serializePoolState("minirepro_pool.json");

    RuntimeMemory::releaseGPU(a);
    RuntimeMemory::releaseGPU(b);
    RuntimeMemory::shutdown();

    std::cout << "minimal allocator repro completed" << std::endl;
    return 0;
}

