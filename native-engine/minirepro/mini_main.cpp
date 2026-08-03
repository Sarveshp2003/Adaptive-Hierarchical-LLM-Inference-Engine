#include "RuntimeMemory.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

int main() {
    RuntimeMemory::initializeGPU(64ULL * 1024 * 1024);
    size_t bsize = 4ULL * 1024 * 1024;
    void* a = RuntimeMemory::allocateGPU(bsize);
    void* b = RuntimeMemory::allocateGPU(bsize);
    std::cout << "Allocated a=" << a << " b=" << b << "\n";

    std::vector<uint8_t> buf(bsize, 0xAA);
    cudaError_t e = cudaMemcpy(b, buf.data(), bsize, cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy result: " << cudaGetErrorString(e) << "\n";
    cudaDeviceSynchronize();

    RuntimeMemory::serializePoolState("out/minirepro_pool.json");
    RuntimeMemory::shutdown();
    std::cout << "Done" << std::endl;
    return 0;
}
