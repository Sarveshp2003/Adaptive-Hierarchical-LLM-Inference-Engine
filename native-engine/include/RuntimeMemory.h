#pragma once
#include <string>
#include <cstddef>

class RuntimeMemory {
public:
    static void initializeGPU(size_t bytes);
    static void* allocateGPU(size_t bytes);
    static void* allocateScratchGPU(size_t bytes);
    static void  releaseGPU(void* ptr);
    static void  shutdown();

    static void serializePoolState(const std::string& path);
    static void attachEventToGPU(void* ptr, void* event); // opaque event pointer

    static void* allocatePinnedHost(size_t bytes);
    static void  releasePinnedHost(void* ptr);
    static bool  registerHost(void* ptr, size_t bytes);
    static bool  unregisterHost(void* ptr);
    static void* getPinnedBuffer(size_t bytes);
    static void  returnPinnedBuffer(void* ptr);
    static void  cleanupPinnedPool();
};
