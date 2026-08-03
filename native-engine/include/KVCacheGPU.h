#ifndef KV_CACHE_GPU_H
#define KV_CACHE_GPU_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

/**
 * GPU-accelerated KV cache management.
 * 
 * Handles:
 * - Transfer KV pages between RAM and GPU VRAM
 * - Manage GPU KV memory allocations
 * - Implement KV cache paging strategy
 */

struct KVCacheGPUPage {
    void* gpuMemPtr;        // GPU memory pointer
    size_t pageSize;        // Size in bytes
    int pageId;             // Logical page ID
    bool isValid;           // Whether data is current
    uint64_t lastUsedToken; // Track for LRU
};

class KVCacheGPU {
private:
    std::vector<std::unique_ptr<KVCacheGPUPage>> pages;
    size_t totalGPUMemory;
    size_t usedGPUMemory;
    size_t maxPages;

    /**
     * Allocate GPU memory for KV page
     */
    void* allocateGPUPage(size_t size);

    /**
     * Free GPU memory page
     */
    void freeGPUPage(void* ptr, size_t size);

    /**
     * Find least recently used page
     */
    int findLRUPage();

public:
    KVCacheGPU(size_t gpuMemoryBytes, size_t pageSize);
    ~KVCacheGPU();

    /**
     * Transfer KV cache from RAM to GPU
     * 
     * @param hostPtr CPU memory pointer
     * @param size Number of bytes
     * @param pageId Logical page identifier
     * @return GPU memory pointer or nullptr on failure
     */
    void* transferToGPU(const void* hostPtr, size_t size, int pageId);

    /**
     * Transfer KV cache from GPU to RAM
     * 
     * @param gpuPtr GPU memory pointer
     * @param hostPtr CPU memory target
     * @param size Number of bytes
     */
    bool transferToHost(void* gpuPtr, void* hostPtr, size_t size);

    /**
     * Get GPU KV page by ID
     */
    KVCacheGPUPage* getPage(int pageId);

    /**
     * Mark page as used (for LRU)
     */
    void markPageUsed(int pageId, uint64_t token);

    /**
     * Evict LRU page if memory full
     */
    bool evictLRUPage();

    /**
     * Get current GPU memory usage
     */
    size_t getUsedMemory() const { return usedGPUMemory; }

    /**
     * Get available GPU memory
     */
    size_t getAvailableMemory() const { return totalGPUMemory - usedGPUMemory; }

    /**
     * Clear all GPU KV pages
     */
    void clear();
};

#endif
