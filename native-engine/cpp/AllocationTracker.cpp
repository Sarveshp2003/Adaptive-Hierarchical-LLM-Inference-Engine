#include "AllocationTracker.h"
#include <sstream>

// Ensure AllocationTracker implementation calls the real cudaMalloc/cudaFree
#ifdef cudaMalloc
#undef cudaMalloc
#endif
#ifdef cudaFree
#undef cudaFree
#endif

namespace AllocationTracker {

static std::mutex g_mutex;
static std::unordered_map<void*, std::string> g_allocs;
static bool g_initialized = false;

void init() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_initialized = true;
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(!g_allocs.empty()) {
        std::cerr << "AllocationTracker: leaked allocations:\n";
        for(auto &kv : g_allocs) {
            std::cerr << "  ptr=" << kv.first << " site=" << kv.second << "\n";
        }
    }
    g_allocs.clear();
    g_initialized = false;
}

cudaError_t trackedCudaMalloc(void** ptr, size_t bytes, const char* site) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    std::lock_guard<std::mutex> lock(g_mutex);
    if(err == cudaSuccess) {
        g_allocs[*ptr] = site ? site : "unknown";
        std::cerr << "[AllocTracker] malloc ptr=" << *ptr << " size=" << bytes << " site=" << (site?site:"?") << "\n";
    } else {
        std::cerr << "[AllocTracker] cudaMalloc failed size=" << bytes << " site=" << (site?site:"?") << " err=" << cudaGetErrorString(err) << "\n";
    }
    return err;
}

cudaError_t trackedCudaFree(void* ptr, const char* site) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(ptr == nullptr) {
        std::cerr << "[AllocTracker] free nullptr site=" << (site?site:"?") << "\n";
        return cudaSuccess;
    }
    auto it = g_allocs.find(ptr);
    if(it == g_allocs.end()) {
        std::cerr << "[AllocTracker] free UNKNOWN ptr=" << ptr << " site=" << (site?site:"?") << "\n";
    } else {
        std::cerr << "[AllocTracker] free ptr=" << ptr << " originally allocated at " << it->second << " site=" << (site?site:"?") << "\n";
        g_allocs.erase(it);
    }
    cudaError_t err = cudaFree(ptr);
    if(err != cudaSuccess) {
        std::cerr << "[AllocTracker] cudaFree returned " << cudaGetErrorString(err) << " for ptr=" << ptr << " site=" << (site?site:"?") << "\n";
    }
    return err;
}

}
