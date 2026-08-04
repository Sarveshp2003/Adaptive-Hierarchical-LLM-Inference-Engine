#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

int main(void) {
    const char* dllPath = "E:\\lib\\adaptive_engine.dll";
    HMODULE h = LoadLibraryA(dllPath);
    if (!h) { printf("LoadLibrary failed: %lu\n", GetLastError()); return 1; }
    
    NativeEngineApi* (*get_api)() = (NativeEngineApi* (*)()) GetProcAddress(h, "adaptive_engine_get_api");
    if (!get_api) { printf("adaptive_engine_get_api not found\n"); FreeLibrary(h); return 2; }
    
    NativeEngineApi* api = get_api(); 
    if (!api) { printf("api returned NULL\n"); FreeLibrary(h); return 3; }
    
    api->start();
    
    int kvPages = api->getKvPages();
    printf("Total KV pages: %d\n", kvPages);
    
    int testsPassed = 0;
    int testsFailed = 0;
    
    // Test 1: moveKvToRam with valid page
    printf("\n--- Test 1: moveKvToRam with valid page ---\n");
    long result = api->moveKvToRam(0);
    if (result > 0) {
        printf("moveKvToRam(0) returned %ld ms (expected > 0)\n", result);
        testsPassed++;
    } else {
        printf("moveKvToRam(0) returned %ld (expected > 0)\n", result);
        testsFailed++;
    }
    
    // Test 2: moveKvToRam with invalid page
    printf("\n--- Test 2: moveKvToRam with invalid page ---\n");
    result = api->moveKvToRam(-1);
    if (result < 0) {
        printf("moveKvToRam(-1) returned %ld (expected < 0)\n", result);
        testsPassed++;
    } else {
        printf("moveKvToRam(-1) returned %ld (expected < 0)\n", result);
        testsFailed++;
    }
    
    // Test 3: moveKvToGpu with valid page
    printf("\n--- Test 3: moveKvToGpu with valid page ---\n");
    result = api->moveKvToGpu(1);
    if (result > 0) {
        printf("moveKvToGpu(1) returned %ld ms (expected > 0)\n", result);
        testsPassed++;
    } else {
        printf("moveKvToGpu(1) returned %ld (expected > 0)\n", result);
        testsFailed++;
    }
    
    // Test 4: compressKv with valid page
    printf("\n--- Test 4: compressKv with valid page ---\n");
    result = api->compressKv(2);
    if (result > 0) {
        printf("compressKv(2) returned %ld ms (expected > 0)\n", result);
        testsPassed++;
    } else {
        printf("compressKv(2) returned %ld (expected > 0)\n", result);
        testsFailed++;
    }
    
    // Test 5: offloadKv with valid page
    printf("\n--- Test 5: offloadKv with valid page ---\n");
    result = api->offloadKv(3);
    if (result > 0) {
        printf("offloadKv(3) returned %ld ms (expected > 0)\n", result);
        testsPassed++;
    } else {
        printf("offloadKv(3) returned %ld (expected > 0)\n", result);
        testsFailed++;
    }
    
    // Test 6: Multiple sequential KV operations
    printf("\n--- Test 6: Sequential KV operations ---\n");
    long r1 = api->moveKvToRam(4);
    long r2 = api->moveKvToGpu(4);
    long r3 = api->compressKv(4);
    if (r1 > 0 && r2 > 0 && r3 > 0) {
        printf("Sequential ops succeeded: moveKvToRam=%ld, moveKvToGpu=%ld, compressKv=%ld\n", r1, r2, r3);
        testsPassed++;
    } else {
        printf("Sequential ops failed\n");
        testsFailed++;
    }
    
    api->stop();
    FreeLibrary(h);
    
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", testsPassed);
    printf("Failed: %d\n", testsFailed);
    
    if (testsFailed == 0) {
        printf("TEST PASSED\n");
        return 0;
    } else {
        printf("TEST FAILED\n");
        return 1;
    }
}
