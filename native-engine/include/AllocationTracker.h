#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <iostream>

namespace AllocationTracker {

void init();
void shutdown();

cudaError_t trackedCudaMalloc(void** ptr, size_t bytes, const char* site);
cudaError_t trackedCudaFree(void* ptr, const char* site);

}

#define TRACKED_CUDA_MALLOC(p, n) AllocationTracker::trackedCudaMalloc((void**)(p),(n), __FILE__ ":" TOSTRING(__LINE__))
#define TRACKED_CUDA_FREE(p) AllocationTracker::trackedCudaFree((void*)(p), __FILE__ ":" TOSTRING(__LINE__))

// Convenience macros to replace cudaMalloc/cudaFree in files that include this header
#ifndef cudaMalloc
#define cudaMalloc(p, n) AllocationTracker::trackedCudaMalloc((void**)(p),(n), __FILE__ ":" TOSTRING(__LINE__))
#endif
#ifndef cudaFree
#define cudaFree(p) AllocationTracker::trackedCudaFree((void*)(p), __FILE__ ":" TOSTRING(__LINE__))
#endif

// Helper stringify macros
#define _TOSTRING(x) #x
#define TOSTRING(x) _TOSTRING(x)
