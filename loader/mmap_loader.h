#pragma once

#include <string>
#include <vector>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#endif

// Mmap-like loader: uses OS memory mapping on Windows, file-buffer fallback otherwise.
class MmapLoader {
public:
    MmapLoader();
    ~MmapLoader();

    // Map the file into memory. Returns false on failure.
    bool mapFile(const std::string &path);

    // Pointer to mapped data (read-only view). nullptr if not mapped.
    const void* data() const;

    // Size of the mapped region in bytes
    size_t size() const;

    // Unmap / release resources
    void unmap();

private:
#ifdef _WIN32
    HANDLE fileHandle_ = INVALID_HANDLE_VALUE;
    HANDLE mappingHandle_ = NULL;
    const void* view_ = nullptr;
    size_t size_ = 0;
    // buffer_ kept for API consistency but unused on Windows mapping path
    std::vector<char> buffer_;
#else
    std::vector<char> buffer_;
#endif
};
