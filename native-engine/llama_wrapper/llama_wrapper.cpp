#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mutex>
#include <fstream>
#include <errno.h>
#include <cstring>

#ifdef HAVE_LLAMA
// When built with the llama submodule, include public headers here (optional)
#include "llama.h"
#endif

#if defined(_WIN32)
#define ADAPTIVE_ENGINE_EXPORT __declspec(dllexport)
#else
#define ADAPTIVE_ENGINE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    typedef struct NativeEngineApi {
        void (*start)();
        void (*stop)();
        long (*prefetchLayer)(int layerId);
        long (*evictLayer)(int layerId);
        long (*keepLayer)(int layerId);
        long (*moveKvToRam)(long kvPageId);
        long (*moveKvToGpu)(long kvPageId);
        long (*compressKv)(long kvPageId);
        long (*offloadKv)(long kvPageId);
        int  (*getCurrentLayer)();
        long (*getGpuMemory)();
        int  (*getKvPages)();
        int  (*getCachedLayers)();
    } NativeEngineApi;

    // A very small wrapper that provides a basic native engine implementation.
    // When llama.cpp or another engine is integrated, replace these with real calls.

#ifdef HAVE_LLAMA
    static struct llama_model * g_model = nullptr;
    static struct llama_context * g_context = nullptr;
    // Per-model cached layers bitmap and mutex to protect it
    static std::vector<char> g_layer_cached;
    // Per-layer pin (refcount) to prevent eviction while in use
    static std::vector<int> g_layer_refcount;
    // Per-layer buffer sizes (in bytes) for real KV tracking
    static std::vector<size_t> g_layer_buffer_sizes;
    // Current layer allocations (kvPageId -> is_allocated)
    static std::vector<char> g_layer_allocated;
    static std::mutex g_cache_mutex;
#endif

    static void lw_start() {
#ifdef HAVE_LLAMA
        const char * path = getenv("LLAMA_MODEL_PATH");
        if (path && !g_model) {
            std::cout << "[llama_wrapper] attempting to load model: " << path << "\n";
            // Verify file exists and is readable
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                std::cerr << "[llama_wrapper] cannot open model file: " << path << " errno=" << errno << " (" << strerror(errno) << ")\n";
            } else {
                auto sz = f.tellg(); f.close();
                std::cout << "[llama_wrapper] model file size=" << sz << " bytes\n";
            }
            struct llama_model_params mparams = llama_model_default_params();
            // Use default params; callers can customize via env or later API
            g_model = llama_model_load_from_file(path, mparams);
            if (g_model) {
                std::cout << "[llama_wrapper] loaded model: " << path << "\n";
                // initialize cache bitmap based on number of model layers
                int nlayer = llama_model_n_layer(g_model);
                std::lock_guard<std::mutex> lk(g_cache_mutex);
                g_layer_cached.assign(nlayer, 0);
                g_layer_refcount.assign(nlayer, 0);
                g_layer_allocated.assign(nlayer, 0);
                
                // Estimate layer buffer sizes (simplified: assume uniform per layer)
                // Real size = (n_head * head_dim * seq_len * 2) * element_size
                // For Llama-3.2-3B: ~2.2GB total KV for context=2048
                // Rough per-layer: ~78MB for 28 layers
                size_t estimated_per_layer = 78 * 1024 * 1024;
                g_layer_buffer_sizes.assign(nlayer, estimated_per_layer);
                
                // Create context for KV cache management
                struct llama_context_params cparams = llama_context_default_params();
                cparams.n_ctx = 1024;  // 1K context window for now
                cparams.n_batch = 256;
                g_context = llama_new_context_with_model(g_model, cparams);
                if (g_context) {
                    std::cout << "[llama_wrapper] created context: n_ctx=" << cparams.n_ctx << " n_batch=" << cparams.n_batch << "\n";
                } else {
                    std::cerr << "[llama_wrapper] warning: could not create context (OOM?), falling back to simulation\n";
                }
            } else {
                std::cerr << "[llama_wrapper] failed to load model: " << path << " -- llama_model_load_from_file returned NULL\n";
            }
        }
#endif
        std::cout << "[llama_wrapper] start()\n";
    }

    static void lw_stop() {
#ifdef HAVE_LLAMA
        if (g_context) {
            llama_free(g_context);
            g_context = nullptr;
        }
        if (g_model) {
            llama_model_free(g_model);
            g_model = nullptr;
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            g_layer_cached.clear();
            g_layer_refcount.clear();
            g_layer_allocated.clear();
            g_layer_buffer_sizes.clear();
            std::cout << "[llama_wrapper] freed model and context\n";
        }
#endif
        std::cout << "[llama_wrapper] stop()\n";
    }

    static long lw_prefetchLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (g_layer_cached[layerId]) return 0; // already cached
            // simulate prefetch cost and mark cached
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            g_layer_cached[layerId] = 1;
            // do not modify refcount; prefetch brings layer into cache but does not pin it
            return 10;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return 5;
    }
    static long lw_evictLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (!g_layer_cached[layerId]) return 0; // already evicted
            if (g_layer_refcount[layerId] > 0) {
                std::cout << "[llama_wrapper] evict denied: layer " << layerId << " is pinned (refcount=" << g_layer_refcount[layerId] << ")\n";
                return -2; // cannot evict while pinned
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            g_layer_cached[layerId] = 0;
            return 2;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static long lw_keepLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            if (layerId < 0 || layerId >= (int)g_layer_cached.size()) return -1;
            if (!g_layer_cached[layerId]) {
                // bring into cache first
                g_layer_cached[layerId] = 1;
            }
            ++g_layer_refcount[layerId];
            return g_layer_refcount[layerId];
        }
#endif
        return 0;
    }
    static long lw_moveKvToRam(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            // Simulate moving buffer to RAM
            // In real implementation: copy from GPU memory to RAM using ggml_backend APIs
            // For now: estimate latency based on buffer size
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            // Assume ~1GB/sec bandwidth: latency = size / 1GB
            long estimated_latency = (buffer_size + 1073741823) / 1073741824;  // Round up to 1ms min
            if (estimated_latency < 1) estimated_latency = 1;
            
            // Mark as allocated in RAM
            g_layer_allocated[kvPageId] = 1;
            
            // Sleep for realistic latency
            std::this_thread::sleep_for(std::chrono::milliseconds(estimated_latency));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] moveKvToRam: kvPageId=" << kvPageId 
                      << " buffer_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return 2;
    }
    static long lw_moveKvToGpu(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            // Simulate moving buffer to GPU
            // In real implementation: use ggml_backend_buffer_copy for GPU memory
            // Assume GPU bandwidth ~2x faster than RAM (~2GB/sec)
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            // latency = size / 2GB
            long estimated_latency = (buffer_size + 2147483647) / 2147483648;
            if (estimated_latency < 1) estimated_latency = 1;
            
            // Mark as NOT allocated in RAM (moved to GPU)
            g_layer_allocated[kvPageId] = 0;
            
            // Sleep for realistic latency
            std::this_thread::sleep_for(std::chrono::milliseconds(estimated_latency));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] moveKvToGpu: kvPageId=" << kvPageId 
                      << " buffer_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return 3;
    }
    static long lw_compressKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            auto start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            
            if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
                return -1;
            }
            
            // Simulate KV compression using quantization
            // Typical compression: F16 -> I8 = 50% reduction
            // Compression speed: ~100MB/sec (CPU bound)
            size_t buffer_size = g_layer_buffer_sizes[kvPageId];
            // latency = size / 100MB
            long estimated_latency = (buffer_size + 104857599) / 104857600;
            if (estimated_latency < 1) estimated_latency = 1;
            
            // Reduce buffer size by 50% (simulate compression)
            g_layer_buffer_sizes[kvPageId] = buffer_size / 2;
            
            // Sleep for realistic latency
            std::this_thread::sleep_for(std::chrono::milliseconds(estimated_latency));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[llama_wrapper] compressKv: kvPageId=" << kvPageId 
                      << " original_size=" << (buffer_size / (1024*1024)) << "MB"
                      << " compressed_size=" << (g_layer_buffer_sizes[kvPageId] / (1024*1024)) << "MB"
                      << " latency=" << latency.count() << "ms\n";
            return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        return 4;
    }
    static long lw_offloadKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
           auto start = std::chrono::high_resolution_clock::now();
           std::lock_guard<std::mutex> lk(g_cache_mutex);
            
           if (kvPageId < 0 || kvPageId >= (long)llama_model_n_layer(g_model)) {
               return -1;
           }
            
           // Simulate offloading buffer (deallocate from device/memory)
           // Remove from both RAM and GPU tracking
           g_layer_allocated[kvPageId] = 0;
            
           // Latency is minimal for deallocation
           std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
           auto end = std::chrono::high_resolution_clock::now();
           auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
           std::cout << "[llama_wrapper] offloadKv: kvPageId=" << kvPageId << " latency=" << latency.count() << "ms\n";
           return latency.count();
        }
#endif
        // Fallback simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static int lw_getCurrentLayer() { return 0; }
    static long lw_getGpuMemory() {
#ifdef HAVE_LLAMA
        if (g_model) {
            // Return model size as proxy for GPU memory requirement
            return (long) llama_model_size(g_model);
        }
#endif
        return 2L * 1024 * 1024 * 1024;
    }
    static int lw_getKvPages() {
#ifdef HAVE_LLAMA
        if (g_model) {
            // No direct mapping; use number of model layers as proxy
            return llama_model_n_layer(g_model);
        }
#endif
        return 256;
    }
    static int lw_getCachedLayers() {
#ifdef HAVE_LLAMA
        if (g_model) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            int count = 0;
            for (char c : g_layer_cached) if (c) ++count;
            return count;
        }
#endif
        return 2;
    }

    static NativeEngineApi g_llama_api = {
        &lw_start,
        &lw_stop,
        &lw_prefetchLayer,
        &lw_evictLayer,
        &lw_keepLayer,
        &lw_moveKvToRam,
        &lw_moveKvToGpu,
        &lw_compressKv,
        &lw_offloadKv,
        &lw_getCurrentLayer,
        &lw_getGpuMemory,
        &lw_getKvPages,
        &lw_getCachedLayers
    };

    ADAPTIVE_ENGINE_EXPORT void adaptive_engine_test_release_layer(int layerId) {
#ifdef HAVE_LLAMA
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        if (layerId < 0 || layerId >= (int)g_layer_refcount.size()) return;
        if (g_layer_refcount[layerId] > 0) --g_layer_refcount[layerId];
        std::cout << "[llama_wrapper] test_release_layer: layer " << layerId << " new_refcount=" << g_layer_refcount[layerId] << "\n";
#endif
    }

    ADAPTIVE_ENGINE_EXPORT NativeEngineApi* adaptive_engine_get_api() {
        return &g_llama_api;
    }
}
