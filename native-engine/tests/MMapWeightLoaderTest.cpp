#include "MMapWeightLoader.h"
#include "Linear.h"
#include "Tensor.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
    struct FileHeader
    {
        char magic[8] = {'A','D','A','P','T','L','M','\0'};
        std::uint32_t version = 1;
        std::uint32_t layerCount = 1;
        std::uint64_t metadataOffset = 0;
        std::uint64_t metadataSize = 0;
        std::uint64_t payloadOffset = 0;
        std::uint64_t payloadSize = 0;
    };

    struct MetadataRecord
    {
        std::uint32_t layerId = 0;
        std::uint32_t weightCount = 0;
        std::uint32_t nameLength = 0;
    };

    struct WeightRecord
    {
        std::uint32_t nameLength = 0;
        std::uint32_t shapeRank = 0;
        std::uint32_t dataType = 0;
        std::uint64_t offset = 0;
        std::uint64_t sizeBytes = 0;
    };

    template <typename T>
    void writeValue(std::ostream& os, T value)
    {
        os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    void writeBytes(std::vector<unsigned char>& out, const T& value)
    {
        const unsigned char* ptr = reinterpret_cast<const unsigned char*>(&value);
        out.insert(out.end(), ptr, ptr + sizeof(T));
    }
}

int main()
{
    const auto path = std::filesystem::temp_directory_path() / "adaptive_mmap_loader_test.bin";
    std::vector<float> qWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> ffnWeights = {5.0f, 6.0f, 7.0f, 8.0f};

    std::vector<unsigned char> metadata;
    std::vector<float> payload;
    payload.insert(payload.end(), qWeights.begin(), qWeights.end());
    payload.insert(payload.end(), ffnWeights.begin(), ffnWeights.end());

    std::vector<unsigned char> layerNameBytes = {'l','a','y','e','r','0'};
    std::vector<unsigned char> qNameBytes = {'a','t','t','n','.','q','_','p','r','o','j'};
    std::vector<unsigned char> ffnNameBytes = {'f','f','n','.','f','c','1'};

    MetadataRecord layerRecord;
    layerRecord.weightCount = 2;
    layerRecord.nameLength = static_cast<std::uint32_t>(layerNameBytes.size());

    metadata.insert(metadata.end(), reinterpret_cast<const unsigned char*>(&layerRecord), reinterpret_cast<const unsigned char*>(&layerRecord) + sizeof(MetadataRecord));
    metadata.insert(metadata.end(), layerNameBytes.begin(), layerNameBytes.end());

    const std::uint64_t metadataSize =
        static_cast<std::uint64_t>(sizeof(MetadataRecord) + layerNameBytes.size() + sizeof(WeightRecord) + qNameBytes.size() + 2 * sizeof(std::uint32_t) + sizeof(WeightRecord) + ffnNameBytes.size() + 2 * sizeof(std::uint32_t));
    const std::uint64_t payloadOffset = static_cast<std::uint64_t>(sizeof(FileHeader)) + metadataSize;

    WeightRecord qRecord;
    qRecord.nameLength = static_cast<std::uint32_t>(qNameBytes.size());
    qRecord.shapeRank = 2;
    qRecord.dataType = 0;
    qRecord.offset = payloadOffset;
    qRecord.sizeBytes = static_cast<std::uint64_t>(qWeights.size() * sizeof(float));

    WeightRecord ffnRecord = qRecord;
    ffnRecord.nameLength = static_cast<std::uint32_t>(ffnNameBytes.size());
    ffnRecord.offset = payloadOffset + static_cast<std::uint64_t>(qWeights.size() * sizeof(float));
    ffnRecord.sizeBytes = static_cast<std::uint64_t>(ffnWeights.size() * sizeof(float));

    metadata.insert(metadata.end(), reinterpret_cast<const unsigned char*>(&qRecord), reinterpret_cast<const unsigned char*>(&qRecord) + sizeof(WeightRecord));
    metadata.insert(metadata.end(), qNameBytes.begin(), qNameBytes.end());
    writeBytes(metadata, std::uint32_t(2));
    writeBytes(metadata, std::uint32_t(2));

    metadata.insert(metadata.end(), reinterpret_cast<const unsigned char*>(&ffnRecord), reinterpret_cast<const unsigned char*>(&ffnRecord) + sizeof(WeightRecord));
    metadata.insert(metadata.end(), ffnNameBytes.begin(), ffnNameBytes.end());
    writeBytes(metadata, std::uint32_t(2));
    writeBytes(metadata, std::uint32_t(2));

    FileHeader header;
    header.metadataOffset = sizeof(FileHeader);
    header.metadataSize = static_cast<std::uint64_t>(metadata.size());
    header.payloadOffset = payloadOffset;
    header.payloadSize = static_cast<std::uint64_t>(payload.size() * sizeof(float));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));
    out.write(reinterpret_cast<const char*>(metadata.data()), metadata.size());
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(float));
    out.close();

    RuntimeMemory::initializeGPU(64ULL * 1024ULL * 1024ULL);
    CUDAStream::initialize();

    MMapWeightLoader loader;
    assert(loader.open(path.string()));
    assert(loader.isOpen());
    assert(loader.loadLayer(0));

    Tensor tensor({2, 2}, DataType::FP32);
    assert(loader.copyWeightToTensor(0, "attn.q_proj", tensor));
    assert(tensor.cpu() != nullptr);
    assert(std::fabs(tensor.cpu()[0] - 1.0f) < 1e-6f);
    assert(std::fabs(tensor.cpu()[3] - 4.0f) < 1e-6f);

    Linear linear(2, 2);
    assert(loader.copyWeightToLinear(0, "ffn.fc1", linear));
    const auto* weights = linear.getWeights().cpu();
    assert(weights != nullptr);
    assert(std::fabs(weights[0] - 5.0f) < 1e-6f);
    assert(std::fabs(weights[3] - 8.0f) < 1e-6f);

    loader.releaseLayer(0);
    CUDAStream::shutdown();
    RuntimeMemory::shutdown();
    std::cout << "mmap_loader_test passed" << std::endl;
    return 0;
}
