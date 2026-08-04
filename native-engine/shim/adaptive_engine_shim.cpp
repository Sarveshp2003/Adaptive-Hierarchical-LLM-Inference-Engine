#include <cstdio>
#include <chrono>
#include <thread>
#include <cstdlib>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
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

    // Simple in-process mock implementation used by CI to validate Phase2 wiring.
    static bool s_started = false;
    static int s_currentLayer = 0;
    static long s_gpuMemory = 2L * 1024 * 1024 * 1024; // 2GB
    static int s_kvPages = 256;
    static int s_cachedLayers = 2;

    static void shim_start() {
        s_started = true;
        std::printf("[adaptive_engine_shim] start() called\n");
    }
    static void shim_stop() {
        s_started = false;
        std::printf("[adaptive_engine_shim] stop() called\n");
    }
    static long shim_prefetchLayer(int layerId) {
        if (!s_started) return -1;
        s_currentLayer = layerId;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::printf("[adaptive_engine_shim] prefetchLayer(%d) -> 5ms\n", layerId);
        return 5;
    }
    static long shim_evictLayer(int layerId) {
        if (!s_started) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::printf("[adaptive_engine_shim] evictLayer(%d) -> 1ms\n", layerId);
        return 1;
    }
    static long shim_keepLayer(int layerId) {
        if (!s_started) return -1;
        return 0;
    }
    static long shim_moveKvToRam(long kvPageId) {
        if (!s_started) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return 2;
    }
    static long shim_moveKvToGpu(long kvPageId) {
        if (!s_started) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return 3;
    }
    static long shim_compressKv(long kvPageId) {
        if (!s_started) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        return 4;
    }
    static long shim_offloadKv(long kvPageId) {
        if (!s_started) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        return 6;
    }
    static int shim_getCurrentLayer() {
        return s_currentLayer;
    }
    static long shim_getGpuMemory() {
        s_gpuMemory += (std::rand() % 100 - 50) * 1024 * 1024;
        return s_gpuMemory;
    }
    static int shim_getKvPages() {
        return s_kvPages;
    }
    static int shim_getCachedLayers() {
        return s_cachedLayers;
    }

    // Exposed function expected by adaptive_scheduler dynamic loader
    EXPORT NativeEngineApi* adaptive_engine_get_api() {
        static NativeEngineApi api = {
            &shim_start,
            &shim_stop,
            &shim_prefetchLayer,
            &shim_evictLayer,
            &shim_keepLayer,
            &shim_moveKvToRam,
            &shim_moveKvToGpu,
            &shim_compressKv,
            &shim_offloadKv,
            &shim_getCurrentLayer,
            &shim_getGpuMemory,
            &shim_getKvPages,
            &shim_getCachedLayers
        };
        return &api;
    }
} // extern "C"
