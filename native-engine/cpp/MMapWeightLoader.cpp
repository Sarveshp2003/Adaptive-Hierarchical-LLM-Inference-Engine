#include "MMapWeightLoader.h"

#include "Tensor.h"
#include "Linear.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace
{
    struct MetadataRecord
    {
        std::uint32_t layerId;
        std::uint32_t weightCount;
        std::uint32_t nameLength;
    };

    struct WeightRecord
    {
        std::uint32_t nameLength;
        std::uint32_t shapeRank;
        std::uint32_t dataType;
        std::uint64_t offset;
        std::uint64_t sizeBytes;
    };

    template <typename T>
    T readValue(const unsigned char* bytes)
    {
        T value{};
        std::memcpy(&value, bytes, sizeof(T));
        return value;
    }

    template <typename T>
    void writeValue(unsigned char* bytes, T value)
    {
        std::memcpy(bytes, &value, sizeof(T));
    }
}

MMapWeightLoader::MMapWeightLoader()
{
}

MMapWeightLoader::~MMapWeightLoader()
{
    if(mappedBase_ != nullptr)
    {
#if defined(_WIN32)
        UnmapViewOfFile(mappedBase_);
        if(mappingHandle_ != nullptr)
        {
            CloseHandle(mappingHandle_);
        }
        if(fileHandle_ != nullptr)
        {
            CloseHandle(fileHandle_);
        }
#else
        munmap(mappedBase_, mappedSize_);
        close(fileDescriptor_);
#endif
    }
}

bool MMapWeightLoader::open(const std::string& path)
{
    if(open_)
    {
        return true;
    }

    filePath_ = path;

#if defined(_WIN32)
    fileHandle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if(fileHandle_ == INVALID_HANDLE_VALUE || fileHandle_ == nullptr)
    {
        return false;
    }

    LARGE_INTEGER fileSize{};
    if(!GetFileSizeEx(reinterpret_cast<HANDLE>(fileHandle_), &fileSize))
    {
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    mappingHandle_ = CreateFileMappingA(reinterpret_cast<HANDLE>(fileHandle_), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if(mappingHandle_ == nullptr)
    {
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    mappedBase_ = MapViewOfFile(reinterpret_cast<HANDLE>(mappingHandle_), FILE_MAP_READ, 0, 0, 0);
    if(mappedBase_ == nullptr)
    {
        CloseHandle(mappingHandle_);
        mappingHandle_ = nullptr;
        CloseHandle(fileHandle_);
        fileHandle_ = nullptr;
        return false;
    }

    mappedSize_ = static_cast<std::size_t>(fileSize.QuadPart);
#else
    fileDescriptor_ = open(path.c_str(), O_RDONLY);
    if(fileDescriptor_ < 0)
    {
        return false;
    }

    struct stat st{};
    if(fstat(fileDescriptor_, &st) != 0)
    {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
        return false;
    }

    mappedSize_ = static_cast<std::size_t>(st.st_size);
    mappedBase_ = mmap(nullptr, mappedSize_, PROT_READ, MAP_PRIVATE, fileDescriptor_, 0);
    if(mappedBase_ == MAP_FAILED)
    {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
        return false;
    }
#endif

    open_ = true;
    return parseMetadata();
}

bool MMapWeightLoader::parseMetadata()
{
    if(mappedBase_ == nullptr || mappedSize_ < sizeof(FileHeader))
    {
        return false;
    }

    const auto* header = reinterpret_cast<const FileHeader*>(mappedBase_);
    if(std::memcmp(header->magic, "ADAPTLM", 7) != 0)
    {
        return false;
    }

    if(header->version != 1)
    {
        return false;
    }

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(mappedBase_);
    const auto* metadata = bytes + header->metadataOffset;
    const auto* payload = bytes + header->payloadOffset;
    (void)payload;

    std::size_t offset = 0;
    layers_.clear();
    layers_.reserve(header->layerCount);

    for(std::uint32_t i = 0; i < header->layerCount; ++i)
    {
        if(offset + sizeof(MetadataRecord) > header->metadataSize)
        {
            layers_.clear();
            return false;
        }

        const auto* record = reinterpret_cast<const MetadataRecord*>(metadata + offset);
        offset += sizeof(MetadataRecord);

        if(record->nameLength == 0)
        {
            layers_.clear();
            return false;
        }

        if(offset + record->nameLength > header->metadataSize)
        {
            layers_.clear();
            return false;
        }

        LayerMetadata layer;
        layer.layerId = static_cast<int>(record->layerId);
        layer.name.assign(reinterpret_cast<const char*>(metadata + offset), record->nameLength);
        offset += record->nameLength;

        layer.weights.reserve(record->weightCount);
        for(std::uint32_t w = 0; w < record->weightCount; ++w)
        {
            if(offset + sizeof(WeightRecord) > header->metadataSize)
            {
                layers_.clear();
                return false;
            }

            const auto* weightRecord = reinterpret_cast<const WeightRecord*>(metadata + offset);
            offset += sizeof(WeightRecord);

            if(offset + weightRecord->nameLength > header->metadataSize)
            {
                layers_.clear();
                return false;
            }

            WeightEntry entry;
            entry.name.assign(reinterpret_cast<const char*>(metadata + offset), weightRecord->nameLength);
            offset += weightRecord->nameLength;
            entry.shape.clear();
            entry.shape.reserve(weightRecord->shapeRank);
            for(std::uint32_t s = 0; s < weightRecord->shapeRank; ++s)
            {
                if(offset + sizeof(std::uint32_t) > header->metadataSize)
                {
                    layers_.clear();
                    return false;
                }
                entry.shape.push_back(static_cast<int>(readValue<std::uint32_t>(metadata + offset)));
                offset += sizeof(std::uint32_t);
            }
            entry.offset = weightRecord->offset;
            entry.sizeBytes = weightRecord->sizeBytes;
            entry.dataType = static_cast<int>(weightRecord->dataType);
            layer.weights.push_back(entry);
        }

        layers_.push_back(layer);
    }

    return true;
}

bool MMapWeightLoader::loadLayer(int layerId)
{
    const LayerMetadata* layer = findLayer(layerId);
    if(layer == nullptr)
    {
        return false;
    }

    for(auto& layerEntry : layers_)
    {
        if(layerEntry.layerId == layerId)
        {
            layerEntry.loaded = true;
            return true;
        }
    }

    return false;
}

void MMapWeightLoader::releaseLayer(int layerId)
{
    for(auto& layerEntry : layers_)
    {
        if(layerEntry.layerId == layerId)
        {
            layerEntry.loaded = false;
            return;
        }
    }
}

const std::vector<LayerMetadata>& MMapWeightLoader::layers() const
{
    return layers_;
}

const LayerMetadata* MMapWeightLoader::findLayer(int layerId) const
{
    for(const auto& layer : layers_)
    {
        if(layer.layerId == layerId)
        {
            return &layer;
        }
    }
    return nullptr;
}

bool MMapWeightLoader::copyWeightToTensor(int layerId, const std::string& weightName, Tensor& tensor) const
{
    if(!open_ || mappedBase_ == nullptr)
    {
        return false;
    }

    const WeightEntry* entry = findWeightEntry(layerId, weightName);
    if(entry == nullptr)
    {
        return false;
    }

    if(entry->dataType != 0)
    {
        return false;
    }

    tensor.allocateCPU();
    const auto expectedElements = static_cast<std::size_t>(tensor.elements());
    const auto expectedBytes = expectedElements * sizeof(float);
    if(entry->sizeBytes != expectedBytes)
    {
        return false;
    }

    const auto* payload = reinterpret_cast<const unsigned char*>(mappedBase_) + entry->offset;
    std::memcpy(tensor.cpu(), payload, entry->sizeBytes);
    return true;
}

bool MMapWeightLoader::copyWeightToLinear(int layerId, const std::string& weightName, Linear& linear) const
{
    if(!open_ || mappedBase_ == nullptr)
    {
        return false;
    }

    const WeightEntry* entry = findWeightEntry(layerId, weightName);
    if(entry == nullptr)
    {
        return false;
    }

    if(entry->dataType != 0)
    {
        return false;
    }

    const auto* payload = reinterpret_cast<const unsigned char*>(mappedBase_) + entry->offset;
    std::vector<float> values(entry->sizeBytes / sizeof(float));
    std::memcpy(values.data(), payload, entry->sizeBytes);
    linear.loadWeights(values);
    return true;
}

bool MMapWeightLoader::isOpen() const
{
    return open_;
}

std::uint64_t MMapWeightLoader::fileSize() const
{
    return static_cast<std::uint64_t>(mappedSize_);
}

const WeightEntry* MMapWeightLoader::findWeightEntry(int layerId, const std::string& weightName) const
{
    const LayerMetadata* layer = findLayer(layerId);
    if(layer == nullptr)
    {
        return nullptr;
    }

    for(const auto& entry : layer->weights)
    {
        if(entry.name == weightName)
        {
            return &entry;
        }
    }
    return nullptr;
}
