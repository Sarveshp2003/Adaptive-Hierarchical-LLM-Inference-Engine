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

// Shim accessor: return current native engine instance. Replaceable later with a dynamic loader that binds to a real NativeEngine.
static MockNativeEngine* getNativeEngine() {
    return MockNativeEngine::instance();
}

// ============================================================================
// ProductionMemoryStateProvider JNI Methods
// ============================================================================

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getCurrentLayerNative
  (JNIEnv *env, jobject obj) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        return engine->currentLayer;
    } catch (const std::exception& e) {
        std::cerr << "[getCurrentLayerNative] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getGpuMemoryNative
  (JNIEnv *env, jobject obj) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        engine->gpuMemoryUsed += (rand() % 100 - 50) * 1024 * 1024;
        return engine->gpuMemoryUsed;
    } catch (const std::exception& e) {
        std::cerr << "[getGpuMemoryNative] Exception: " << e.what() << "\n";
        return 2L * 1024 * 1024 * 1024;
    }
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getKvPagesNative
  (JNIEnv *env, jobject obj) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        return engine->kvPages;
    } catch (const std::exception& e) {
        std::cerr << "[getKvPagesNative] Exception: " << e.what() << "\n";
        return 256;
    }
}

JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getCachedLayersNative
  (JNIEnv *env, jobject obj) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        return engine->cachedLayers;
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        engine->start();
        std::cout << "[Phase2NativeEngineAdapter] Engine started via JNI\n";
    } catch (const std::exception& e) {
        std::cerr << "[nativeStart] Exception: " << e.what() << "\n";
    }
}

JNIEXPORT void JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeStop
  (JNIEnv *env, jobject obj, jobject nativeEngine) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        engine->stop();
        std::cout << "[Phase2NativeEngineAdapter] Engine stopped via JNI\n";
    } catch (const std::exception& e) {
        std::cerr << "[nativeStop] Exception: " << e.what() << "\n";
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativePrefetchLayer
  (JNIEnv *env, jobject obj, jobject nativeEngine, jint layerId) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->prefetchLayer(layerId);
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->evictLayer(layerId);
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->keepLayer(layerId);
        return latency;
    } catch (const std::exception& e) {
        std::cerr << "[nativeKeepLayer] Exception: " << e.what() << "\n";
        return -1;
    }
}

JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeMoveKvToRam
  (JNIEnv *env, jobject obj, jobject nativeEngine, jlong kvPageId) {
    try {
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->moveKvToRam(kvPageId);
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->moveKvToGpu(kvPageId);
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->compressKv(kvPageId);
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
        MockNativeEngine* engine = MockNativeEngine::instance();
        long latency = engine->offloadKv(kvPageId);
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
