#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>

#ifdef HAVE_LLAMA
// When built with the llama submodule, include public headers here (optional)
#include "llama.h"
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
#endif

    static void lw_start() {
#ifdef HAVE_LLAMA
        const char * path = getenv("LLAMA_MODEL_PATH");
        if (path && !g_model) {
            std::cout << "[llama_wrapper] attempting to load model: " << path << "\n";
            struct llama_model_params mparams = llama_model_default_params();
            // Use default params; callers can customize via env or later API
            g_model = llama_model_load_from_file(path, mparams);
            if (g_model) {
                std::cout << "[llama_wrapper] loaded model: " << path << "\n";
            } else {
                std::cerr << "[llama_wrapper] failed to load model: " << path << "\n";
            }
        }
#endif
        std::cout << "[llama_wrapper] start()\n";
    }

    static void lw_stop() {
#ifdef HAVE_LLAMA
        if (g_model) {
            llama_model_free(g_model);
            g_model = nullptr;
            std::cout << "[llama_wrapper] freed model\n";
        }
#endif
        std::cout << "[llama_wrapper] stop()\n";
    }

    static long lw_prefetchLayer(int layerId) {
#ifdef HAVE_LLAMA
        // Prefetch could map to loading a layer into RAM; llama.cpp handles this internally when using llama_model_load_from_file.
        if (g_model) {
            // No-op for now, return small latency
            return 5;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return 5;
    }
    static long lw_evictLayer(int layerId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            // No direct API to evict a single layer; return simulated latency
            return 1;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static long lw_keepLayer(int layerId) {
        return 0;
    }
    static long lw_moveKvToRam(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            // No-op mapping; simulate small latency
            return 2;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return 2;
    }
    static long lw_moveKvToGpu(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            // If GPU offload supported, this could call into ggml backend to move buffers.
            return 3;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return 3;
    }
    static long lw_compressKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            // Could invoke quantization routines; simulated here
            return 4;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        return 4;
    }
    static long lw_offloadKv(long kvPageId) {
#ifdef HAVE_LLAMA
        if (g_model) {
            return 6;
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        return 6;
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
            return 2;
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

    NativeEngineApi* adaptive_engine_get_api() {
        return &g_llama_api;
    }
}
