#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

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

typedef void (*release_t)(int layerId);

int main(void) {
    const char* dllPath = "E:\\lib\\adaptive_engine.dll";
    HMODULE h = LoadLibraryA(dllPath);
    if (!h) { printf("LoadLibrary failed: %lu\n", GetLastError()); return 1; }
    NativeEngineApi* (*get_api)() = (NativeEngineApi* (*)()) GetProcAddress(h, "adaptive_engine_get_api");
    release_t release = (release_t) GetProcAddress(h, "adaptive_engine_test_release_layer");
    if (!get_api) { printf("adaptive_engine_get_api not found\n"); FreeLibrary(h); return 2; }
    if (!release) { printf("adaptive_engine_test_release_layer not found\n"); FreeLibrary(h); return 3; }
    NativeEngineApi* api = get_api(); if (!api) { printf("api returned NULL\n"); FreeLibrary(h); return 4; }
    api->start();
    // ensure layer 0 prefetched
    api->prefetchLayer(0);
    // pin layer 0
    long pin = api->keepLayer(0);
    printf("pin returned %ld\n", pin);
    // attempt evict - should be denied (-2)
    long ev = api->evictLayer(0);
    printf("evict returned %ld (expected negative denial)\n", ev);
    if (ev != -2) { printf("expected evict denied with -2\n"); api->stop(); FreeLibrary(h); return 5; }
    // release pin via test hook
    release(0);
    // now evict should succeed (>0)
    ev = api->evictLayer(0);
    printf("evict after release returned %ld\n", ev);
    if (ev <= 0) { printf("expected successful evict >0\n"); api->stop(); FreeLibrary(h); return 6; }
    api->stop(); FreeLibrary(h);
    printf("TEST PASSED\n");
    return 0;
}
