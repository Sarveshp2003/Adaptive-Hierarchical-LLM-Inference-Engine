#include "KVCacheGPU.h"
#include "Logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

KVCacheGPU::KVCacheGPU(size_t gpuMemoryBytes, size_t pageSize)
    : totalGPUMemory(gpuMemoryBytes), 
      usedGPUMemory(0),
      maxPages(gpuMemoryBytes / pageSize) {
    
    LOG_INFO("KVCacheGPU initialized: " + std::to_string(gpuMemoryBytes / (1024*1024)) + 
             "MB, max pages=" + std::to_string(maxPages));
}

KVCacheGPU::~KVCacheGPU() {
    clear();
}

void* KVCacheGPU::allocateGPUPage(size_t size) {
    if (usedGPUMemory + size > totalGPUMemory) {
        LOG_WARN("GPU memory insufficient: need " + std::to_string(size) + 
                " but only " + std::to_string(totalGPUMemory - usedGPUMemory) + " available");
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(err)));
        return nullptr;
    }

    usedGPUMemory += size;
    LOG_DEBUG("Allocated GPU page: " + std::to_string(size) + " bytes at " + 
              std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    
    return ptr;
}

void KVCacheGPU::freeGPUPage(void* ptr, size_t size) {
    if (!ptr) return;

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to free GPU memory: " + std::string(cudaGetErrorString(err)));
        return;
    }

    usedGPUMemory -= size;
    LOG_DEBUG("Freed GPU page: " + std::to_string(size) + " bytes");
}

int KVCacheGPU::findLRUPage() {
    int lruPageId = -1;
    uint64_t minUsedToken = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < pages.size(); i++) {
        if (pages[i] && pages[i]->lastUsedToken < minUsedToken) {
            minUsedToken = pages[i]->lastUsedToken;
            lruPageId = i;
        }
    }

    return lruPageId;
}

void* KVCacheGPU::transferToGPU(const void* hostPtr, size_t size, int pageId) {
    // Try to allocate GPU memory
    void* gpuPtr = allocateGPUPage(size);
    if (!gpuPtr) {
        // Evict LRU page and retry
        if (evictLRUPage()) {
            gpuPtr = allocateGPUPage(size);
        }
        if (!gpuPtr) {
            LOG_ERROR("Cannot allocate GPU memory for page " + std::to_string(pageId));
            return nullptr;
        }
    }

    // Transfer data from host to GPU
    cudaError_t err = cudaMemcpy(gpuPtr, hostPtr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to copy to GPU: " + std::string(cudaGetErrorString(err)));
        freeGPUPage(gpuPtr, size);
        return nullptr;
    }

    // Create page metadata
    auto page = std::make_unique<KVCacheGPUPage>();
    page->gpuMemPtr = gpuPtr;
    page->pageSize = size;
    page->pageId = pageId;
    page->isValid = true;
    page->lastUsedToken = 0;

    // Ensure pages vector is large enough
    if (pageId >= (int)pages.size()) {
        pages.resize(pageId + 1);
    }

    pages[pageId] = std::move(page);

    LOG_DEBUG("Transferred page " + std::to_string(pageId) + " to GPU: " + 
              std::to_string(size) + " bytes");

    return gpuPtr;
}

bool KVCacheGPU::transferToHost(void* gpuPtr, void* hostPtr, size_t size) {
    cudaError_t err = cudaMemcpy(hostPtr, gpuPtr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to copy from GPU: " + std::string(cudaGetErrorString(err)));
        return false;
    }

    LOG_DEBUG("Transferred " + std::to_string(size) + " bytes from GPU to host");
    return true;
}

KVCacheGPUPage* KVCacheGPU::getPage(int pageId) {
    if (pageId < 0 || pageId >= (int)pages.size()) {
        return nullptr;
    }
    return pages[pageId].get();
}

void KVCacheGPU::markPageUsed(int pageId, uint64_t token) {
    auto page = getPage(pageId);
    if (page) {
        page->lastUsedToken = token;
    }
}

bool KVCacheGPU::evictLRUPage() {
    int lruPageId = findLRUPage();
    if (lruPageId < 0) {
        return false;
    }

    auto page = pages[lruPageId].get();
    if (!page || !page->gpuMemPtr) {
        return false;
    }

    freeGPUPage(page->gpuMemPtr, page->pageSize);
    pages[lruPageId] = nullptr;

    LOG_INFO("Evicted LRU page " + std::to_string(lruPageId));
    return true;
}

void KVCacheGPU::clear() {
    for (auto& page : pages) {
        if (page && page->gpuMemPtr) {
            freeGPUPage(page->gpuMemPtr, page->pageSize);
        }
    }
    pages.clear();
    usedGPUMemory = 0;
    LOG_INFO("KVCacheGPU cleared");
}
