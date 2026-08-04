#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>

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

    static void lw_start() {
        std::cout << "[llama_wrapper] start()\\n";
    }
    static void lw_stop() {
        std::cout << "[llama_wrapper] stop()\\n";
    }
    static long lw_prefetchLayer(int layerId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return 5;
    }
    static long lw_evictLayer(int layerId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 1;
    }
    static long lw_keepLayer(int layerId) {
        return 0;
    }
    static long lw_moveKvToRam(long kvPageId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return 2;
    }
    static long lw_moveKvToGpu(long kvPageId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return 3;
    }
    static long lw_compressKv(long kvPageId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        return 4;
    }
    static long lw_offloadKv(long kvPageId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        return 6;
    }
    static int lw_getCurrentLayer() { return 0; }
    static long lw_getGpuMemory() { return 2L * 1024 * 1024 * 1024; }
    static int lw_getKvPages() { return 256; }
    static int lw_getCachedLayers() { return 2; }

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
