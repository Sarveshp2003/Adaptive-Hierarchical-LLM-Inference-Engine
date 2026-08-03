#ifndef MMAP_WEIGHT_LOADER_H
#define MMAP_WEIGHT_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

class Tensor;
class Linear;

struct WeightEntry
{
    std::string name;
    std::vector<int> shape;
    std::uint64_t offset = 0;
    std::uint64_t sizeBytes = 0;
    int dataType = 0;
};

struct LayerMetadata
{
    int layerId = -1;
    std::string name;
    std::vector<WeightEntry> weights;
    bool loaded = false;
};

class MMapWeightLoader
{
public:
    MMapWeightLoader();
    ~MMapWeightLoader();

    bool open(const std::string& path);
    bool loadLayer(int layerId);
    void releaseLayer(int layerId);

    const std::vector<LayerMetadata>& layers() const;
    const LayerMetadata* findLayer(int layerId) const;

    bool copyWeightToTensor(int layerId, const std::string& weightName, Tensor& tensor) const;
    bool copyWeightToLinear(int layerId, const std::string& weightName, Linear& linear) const;

    bool isOpen() const;
    std::uint64_t fileSize() const;

private:
    struct FileHeader
    {
        char magic[8];
        std::uint32_t version;
        std::uint32_t layerCount;
        std::uint64_t metadataOffset;
        std::uint64_t metadataSize;
        std::uint64_t payloadOffset;
        std::uint64_t payloadSize;
    };

    bool parseMetadata();
    const WeightEntry* findWeightEntry(int layerId, const std::string& weightName) const;

    std::string filePath_;
    void* mappedBase_ = nullptr;
    std::size_t mappedSize_ = 0;
    std::vector<LayerMetadata> layers_;
    bool open_ = false;

#if defined(_WIN32)
    void* fileHandle_ = nullptr;
    void* mappingHandle_ = nullptr;
#else
    int fileDescriptor_ = -1;
#endif
};

#endif
