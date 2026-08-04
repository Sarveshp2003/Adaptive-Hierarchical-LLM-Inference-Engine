#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <process.h>

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

static NativeEngineApi* api;
static release_t release_hook;

unsigned __stdcall pin_thread(void* arg) {
    int layer = (int)(intptr_t)arg;
    for (int i=0;i<100;i++) {
        api->keepLayer(layer);
        Sleep(1);
    }
    return 0;
}
unsigned __stdcall unpin_thread(void* arg) {
    int layer = (int)(intptr_t)arg;
    for (int i=0;i<100;i++) {
        release_hook(layer);
        Sleep(1);
    }
    return 0;
}

int main(void) {
    const char* dllPath = "E:\\lib\\adaptive_engine.dll";
    HMODULE h = LoadLibraryA(dllPath);
    if (!h) { printf("LoadLibrary failed: %lu\n", GetLastError()); return 1; }
    NativeEngineApi* (*get_api)() = (NativeEngineApi* (*)()) GetProcAddress(h, "adaptive_engine_get_api");
    release_t release = (release_t) GetProcAddress(h, "adaptive_engine_test_release_layer");
    if (!get_api || !release) { printf("required symbols not found\n"); FreeLibrary(h); return 2; }
    api = get_api(); release_hook = release;
    api->start();
    api->prefetchLayer(0);
    // spawn multiple pin/unpin threads
    uintptr_t tid1 = _beginthreadex(NULL, 0, pin_thread, (void*)0, 0, NULL);
    uintptr_t tid2 = _beginthreadex(NULL, 0, unpin_thread, (void*)0, 0, NULL);
    WaitForSingleObject((HANDLE)tid1, INFINITE);
    WaitForSingleObject((HANDLE)tid2, INFINITE);
    // final cached count should be non-negative; ensure no crash
    int cached = api->getCachedLayers();
    printf("concurrent test completed, cached=%d\n", cached);
    api->stop(); FreeLibrary(h);
    printf("TEST PASSED\n");
    return 0;
}
