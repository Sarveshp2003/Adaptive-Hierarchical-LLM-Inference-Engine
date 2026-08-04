#include <jni.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>

// Include generated JNI headers
#include "headers/com_adaptivellm_scheduler_ProductionMemoryStateProvider.h"
#include "headers/com_adaptivellm_scheduler_Phase2NativeEngineAdapter.h"

// Forward declarations for native engine access
class MockNativeEngine {
public:
    int currentLayer = 0;
    long gpuMemoryUsed = 2L * 1024 * 1024 * 1024;  // 2GB default
    int kvPages = 256;
    int cachedLayers = 2;
    bool running = false;

    static MockNativeEngine* instance() {
        static MockNativeEngine engine;
        return &engine;
    }

    void start() {
        running = true;
        std::cout << "[MockNativeEngine] Started\n";
    }

    void stop() {
        running = false;
        std::cout << "[MockNativeEngine] Stopped\n";
    }

    long prefetchLayer(int layerId) {
        if (!running) return -1;
        currentLayer = layerId;
        // Simulate prefetch latency: 5-10ms
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }

    long evictLayer(int layerId) {
        if (!running) return -1;
        // Eviction is faster: 1-3ms
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }

    long keepLayer(int layerId) {
        if (!running) return -1;
        return 0;  // No-op
    }

    long moveKvToRam(long kvPageId) {
        if (!running) return -1;
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }

    long moveKvToGpu(long kvPageId) {
        if (!running) return -1;
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }

    long compressKv(long kvPageId) {
        if (!running) return -1;
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }

    long offloadKv(long kvPageId) {
        if (!running) return -1;
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return latency.count();
    }
};

// Native engine C API interface (optional real engine). The code will attempt to dynamically
// load a native engine library that exposes `adaptive_engine_get_api()` returning a
// pointer to this struct. If the library or symbol is not available, the implementation
// falls back to the in-process MockNativeEngine above.

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

    typedef NativeEngineApi* (*GetApiFn)();
}

static NativeEngineApi* g_nativeApi = nullptr;
static void* g_nativeHandle = nullptr;

#ifdef _WIN32
#include <windows.h>
static void tryLoadNativeApi() {
    if (g_nativeApi) return;
    const char* candidates[] = { "adaptive_engine.dll", "native_engine.dll", "adaptive_scheduler_engine.dll" };

    // First try loading engines from the same directory as this DLL (common when java.library.path is used)
    HMODULE self = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&tryLoadNativeApi, &self)) {
        CHAR modulePath[MAX_PATH];
        if (GetModuleFileNameA(self, modulePath, MAX_PATH)) {
            std::string dir(modulePath);
            auto pos = dir.find_last_of("\\/");
            if (pos != std::string::npos) dir = dir.substr(0, pos+1);
            else dir.clear();

            for (auto name : candidates) {
                std::string full = dir + name;
                HMODULE h = LoadLibraryA(full.c_str());
                if (!h) continue;
                auto sym = (FARPROC)GetProcAddress(h, "adaptive_engine_get_api");
                if (!sym) { FreeLibrary(h); continue; }
                auto getApi = (GetApiFn)sym;
                g_nativeApi = getApi();
                if (g_nativeApi) { g_nativeHandle = (void*)h; std::cout << "[NativeLoader] Loaded native engine: " << full << "\n"; return; }
                FreeLibrary(h);
            }
        }
    }

    // Fall back to system search paths
    for (auto name : candidates) {
        HMODULE h = LoadLibraryA(name);
        if (!h) continue;
        auto sym = (FARPROC)GetProcAddress(h, "adaptive_engine_get_api");
        if (!sym) { FreeLibrary(h); continue; }
        auto getApi = (GetApiFn)sym;
        g_nativeApi = getApi();
        if (g_nativeApi) { g_nativeHandle = (void*)h; std::cout << "[NativeLoader] Loaded native engine: " << name << "\n"; return; }
        FreeLibrary(h);
    }

    std::cout << "[NativeLoader] No native engine library found; falling back to MockNativeEngine\n";
}
#else
#include <dlfcn.h>
static void tryLoadNativeApi() {
    if (g_nativeApi) return;
    const char* candidates[] = { "libadaptive_engine.so", "libnative_engine.so", "libadaptive_scheduler_engine.so" };
    for (auto name : candidates) {
        void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto sym = dlsym(h, "adaptive_engine_get_api");
        if (!sym) { dlclose(h); continue; }
        auto getApi = (GetApiFn)sym;
        g_nativeApi = getApi();
        if (g_nativeApi) { g_nativeHandle = h; std::cout << "[NativeLoader] Loaded native engine: " << name << "\n"; return; }
        dlclose(h);
    }
    std::cout << "[NativeLoader] No native engine library found; falling back to MockNativeEngine\n";
}
#endif

// Helper wrappers that call either the loaded native API or the MockNativeEngine fallback.
static void native_start() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->start) { g_nativeApi->start(); return; }
    MockNativeEngine::instance()->start();
}
static void native_stop() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->stop) { g_nativeApi->stop(); return; }
    MockNativeEngine::instance()->stop();
}
static long native_prefetchLayer(int layerId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->prefetchLayer) return g_nativeApi->prefetchLayer(layerId);
    return MockNativeEngine::instance()->prefetchLayer(layerId);
}
static long native_evictLayer(int layerId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->evictLayer) return g_nativeApi->evictLayer(layerId);
    return MockNativeEngine::instance()->evictLayer(layerId);
}
static long native_keepLayer(int layerId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->keepLayer) return g_nativeApi->keepLayer(layerId);
    return MockNativeEngine::instance()->keepLayer(layerId);
}
static long native_moveKvToRam(long kvPageId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->moveKvToRam) return g_nativeApi->moveKvToRam(kvPageId);
    return MockNativeEngine::instance()->moveKvToRam(kvPageId);
}
static long native_moveKvToGpu(long kvPageId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->moveKvToGpu) return g_nativeApi->moveKvToGpu(kvPageId);
    return MockNativeEngine::instance()->moveKvToGpu(kvPageId);
}
static long native_compressKv(long kvPageId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->compressKv) return g_nativeApi->compressKv(kvPageId);
    return MockNativeEngine::instance()->compressKv(kvPageId);
}
static long native_offloadKv(long kvPageId) {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->offloadKv) return g_nativeApi->offloadKv(kvPageId);
    return MockNativeEngine::instance()->offloadKv(kvPageId);
}
static int native_getCurrentLayer() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->getCurrentLayer) return g_nativeApi->getCurrentLayer();
    return MockNativeEngine::instance()->currentLayer;
}
static long native_getGpuMemory() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->getGpuMemory) return g_nativeApi->getGpuMemory();
    // Add small jitter when using mock
    MockNativeEngine* m = MockNativeEngine::instance();
    m->gpuMemoryUsed += (rand() % 100 - 50) * 1024 * 1024;
    return m->gpuMemoryUsed;
}
static int native_getKvPages() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->getKvPages) return g_nativeApi->getKvPages();
    return MockNativeEngine::instance()->kvPages;
}
static int native_getCachedLayers() {
    tryLoadNativeApi();
    if (g_nativeApi && g_nativeApi->getCachedLayers) return g_nativeApi->getCachedLayers();
    return MockNativeEngine::instance()->cachedLayers;
}

// ============================================================================
// ProductionMemoryStateProvider JNI Methods
// ============================================================================

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getCurrentLayerNative
  (JNIEnv *env, jobject obj) {
    try {
        return native_getCurrentLayer();
    } catch (const std::exception& e) {
        std::cerr << "[getCurrentLayerNative] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getGpuMemoryNative
  (JNIEnv *env, jobject obj) {
    try {
        return native_getGpuMemory();
    } catch (const std::exception& e) {
        std::cerr << "[getGpuMemoryNative] Exception: " << e.what() << "\n";
        return 2L * 1024 * 1024 * 1024;
    }
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getKvPagesNative
  (JNIEnv *env, jobject obj) {
    try {
        return native_getKvPages();
    } catch (const std::exception& e) {
        std::cerr << "[getKvPagesNative] Exception: " << e.what() << "\n";
        return 256;
    }
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getCachedLayersNative
  (JNIEnv *env, jobject obj) {
    try {
        return native_getCachedLayers();
    } catch (const std::exception& e) {
        std::cerr << "[getCachedLayersNative] Exception: " << e.what() << "\n";
        return 2;
    }
}

// ============================================================================
// Phase2NativeEngineAdapter JNI Methods
// ============================================================================

JNIEXPORT void JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeStart
  (JNIEnv *env, jobject obj, jobject nativeEngine) {
    try {
        native_start();
        std::cout << "[Phase2NativeEngineAdapter] Engine started via JNI\n";
    } catch (const std::exception& e) {
        std::cerr << "[nativeStart] Exception: " << e.what() << "\n";
    }
}

JNIEXPORT void JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeStop
  (JNIEnv *env, jobject obj, jobject nativeEngine) {
    try {
        native_stop();
        std::cout << "[Phase2NativeEngineAdapter] Engine stopped via JNI\n";
    } catch (const std::exception& e) {
        std::cerr << "[nativeStop] Exception: " << e.what() << "\n";
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativePrefetchLayer
  (JNIEnv *env, jobject obj, jobject nativeEngine, jint layerId) {
    try {
        long latency = native_prefetchLayer(layerId);
        std::cout << "[nativePrefetchLayer] Layer " << layerId << " prefetched (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativePrefetchLayer] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeEvictLayer
  (JNIEnv *env, jobject obj, jobject nativeEngine, jint layerId) {
    try {
        long latency = native_evictLayer(layerId);
        std::cout << "[nativeEvictLayer] Layer " << layerId << " evicted (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeEvictLayer] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeKeepLayer
  (JNIEnv *env, jobject obj, jobject nativeEngine, jint layerId) {
    try {
        long latency = native_keepLayer(layerId);
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeKeepLayer] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeMoveKvToRam
  (JNIEnv *env, jobject obj, jobject nativeEngine, jlong kvPageId) {
    try {
        long latency = native_moveKvToRam(kvPageId);
        std::cout << "[nativeMoveKvToRam] KV page " << kvPageId << " moved (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeMoveKvToRam] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeMoveKvToGpu
  (JNIEnv *env, jobject obj, jobject nativeEngine, jlong kvPageId) {
    try {
        long latency = native_moveKvToGpu(kvPageId);
        std::cout << "[nativeMoveKvToGpu] KV page " << kvPageId << " moved (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeMoveKvToGpu] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeCompressKv
  (JNIEnv *env, jobject obj, jobject nativeEngine, jlong kvPageId) {
    try {
        long latency = native_compressKv(kvPageId);
        std::cout << "[nativeCompressKv] KV page " << kvPageId << " compressed (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeCompressKv] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeOffloadKv
  (JNIEnv *env, jobject obj, jobject nativeEngine, jlong kvPageId) {
    try {
        long latency = native_offloadKv(kvPageId);
        std::cout << "[nativeOffloadKv] KV page " << kvPageId << " offloaded (latency=" << latency << "ms)\n";
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeOffloadKv] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    std::cout << "[JNI_OnLoad] adaptive_scheduler native library loaded successfully\n";
    return JNI_VERSION_11;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    std::cout << "[JNI_OnUnload] adaptive_scheduler native library unloaded\n";
}
