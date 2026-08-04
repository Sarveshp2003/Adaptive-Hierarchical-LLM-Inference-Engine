#include <windows.h>
#include <stdio.h>

int main(void) {
    const char* dllPath = "E:\\lib\\adaptive_engine.dll";
    HMODULE h = LoadLibraryA(dllPath);
    if (!h) {
        printf("LoadLibrary failed: %lu\n", GetLastError());
        return 1;
    }
    FARPROC f = GetProcAddress(h, "adaptive_engine_get_api");
    if (!f) {
        printf("Symbol adaptive_engine_get_api not found\n");
        FreeLibrary(h);
        return 2;
    }
    printf("Symbol adaptive_engine_get_api found at %p\n", (void*)f);
    FreeLibrary(h);
    return 0;
}
