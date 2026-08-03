#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

int main() {
    const char *path_env = std::getenv("PATH");
    if (!path_env) {
        std::cerr << "PATH not set\n";
        return 2;
    }
    std::string pathstr(path_env);
    std::vector<std::filesystem::path> dirs;
    size_t start = 0;
    while (start < pathstr.size()) {
        size_t pos = pathstr.find(';', start);
        if (pos == std::string::npos) pos = pathstr.size();
        std::string part = pathstr.substr(start, pos - start);
        if (!part.empty()) dirs.emplace_back(part);
        start = pos + 1;
    }

    std::filesystem::path found;
    for (auto &d : dirs) {
        try {
            if (!std::filesystem::exists(d)) continue;
            for (auto &entry : std::filesystem::directory_iterator(d)) {
                auto name = entry.path().filename().string();
                if (name.rfind("cudart64_", 0) == 0 && entry.path().extension() == ".dll") {
                    found = entry.path();
                    break;
                }
            }
        } catch (...) {}
        if (!found.empty()) break;
    }

    if (found.empty()) {
        std::cout << "No cudart DLL found in PATH" << std::endl;
        return 1;
    }

    std::wcout << L"Found cudart DLL: " << found.wstring() << std::endl;
    HMODULE h = LoadLibraryW(found.wstring().c_str());
    if (!h) {
        std::cerr << "LoadLibrary failed: " << GetLastError() << std::endl;
        return 3;
    }

    typedef int (*cudaGetDeviceCount_t)(int*);
    typedef const char* (*cudaGetErrorString_t)(int);

    auto pCount = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
    auto pErrStr = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
    if (!pCount || !pErrStr) {
        std::cerr << "Failed to find CUDA symbols in cudart: " << GetLastError() << std::endl;
        FreeLibrary(h);
        return 4;
    }

    int devCount = 0;
    int err = pCount(&devCount);
    if (err != 0) {
        const char *s = pErrStr(err);
        std::cout << "cudaGetDeviceCount error: " << (s? s: "(null)") << " (code " << err << ")\n";
    } else {
        std::cout << "cudaGetDeviceCount reports " << devCount << " device(s)\n";
    }

    FreeLibrary(h);
    return 0;
}
