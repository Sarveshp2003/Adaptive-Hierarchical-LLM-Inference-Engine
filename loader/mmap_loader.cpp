#include "mmap_loader.h"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <cstring>
#endif

MmapLoader::MmapLoader() = default;

MmapLoader::~MmapLoader() {
    unmap();
}

bool MmapLoader::mapFile(const std::string &path) {
#ifdef _WIN32
    unmap();
    // Open file
    fileHandle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "mmap_loader (win): CreateFileA failed for: " << path << "\n";
        return false;
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(fileHandle_, &liSize)) {
        std::cerr << "mmap_loader (win): GetFileSizeEx failed\n";
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
        return false;
    }
    size_ = static_cast<size_t>(liSize.QuadPart);
    if (size_ == 0) return false;

    mappingHandle_ = CreateFileMappingA(fileHandle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mappingHandle_) {
        std::cerr << "mmap_loader (win): CreateFileMapping failed\n";
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    view_ = MapViewOfFile(mappingHandle_, FILE_MAP_READ, 0, 0, 0);
    if (!view_) {
        std::cerr << "mmap_loader (win): MapViewOfFile failed\n";
        CloseHandle(mappingHandle_);
        mappingHandle_ = NULL;
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    return true;
#else
    buffer_.clear();
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "mmap_loader: failed to open file: " << path << "\n";
        return false;
    }
    std::streamsize sz = in.tellg();
    if (sz <= 0) return false;
    in.seekg(0, std::ios::beg);
    buffer_.resize(static_cast<size_t>(sz));
    if (!in.read(buffer_.data(), sz)) {
        std::cerr << "mmap_loader: failed to read file: " << path << "\n";
        buffer_.clear();
        return false;
    }
    return true;
#endif
}

const void* MmapLoader::data() const {
#ifdef _WIN32
    return view_ ? view_ : nullptr;
#else
    return buffer_.empty() ? nullptr : static_cast<const void*>(buffer_.data());
#endif
}

size_t MmapLoader::size() const {
#ifdef _WIN32
    return size_;
#else
    return buffer_.size();
#endif
}

void MmapLoader::unmap() {
#ifdef _WIN32
    if (view_) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (mappingHandle_) {
        CloseHandle(mappingHandle_);
        mappingHandle_ = NULL;
    }
    if (fileHandle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
    }
    size_ = 0;
    buffer_.clear();
#else
    buffer_.clear();
    buffer_.shrink_to_fit();
#endif
}
