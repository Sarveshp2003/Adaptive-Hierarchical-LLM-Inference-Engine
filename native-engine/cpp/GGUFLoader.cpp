#include "GGUFLoader.h"
#include "Logger.h"
#include <cstring>
#include <algorithm>

GGUFLoader::GGUFLoader() {
    header.magic = 0;
    header.version = 0;
    header.tensorCount = 0;
    header.tensorDataOffset = 0;
    header.metadataKvCount = 0;
}

GGUFLoader::~GGUFLoader() {
    tensors.clear();
    metadata.clear();
}

bool GGUFLoader::loadMetadata(const std::string& path) {
    filePath = path;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERROR("Failed to open GGUF file: " + path);
        return false;
    }

    // determine file size for defensive checks (use 64-bit APIs where available)
#if defined(_WIN32)
    if(_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); LOG_ERROR("Failed to seek GGUF file"); return false; }
    __int64 szi = _ftelli64(f);
    if(szi < 0) { fclose(f); LOG_ERROR("Failed to determine GGUF file size"); return false; }
    fileSize_ = static_cast<uint64_t>(szi);
    _fseeki64(f, 0, SEEK_SET);
#else
    if(fseek(f, 0, SEEK_END) != 0) { fclose(f); LOG_ERROR("Failed to seek GGUF file"); return false; }
    long sz = ftell(f);
    if(sz < 0) { fclose(f); LOG_ERROR("Failed to determine GGUF file size"); return false; }
    fileSize_ = static_cast<uint64_t>(sz);
    rewind(f);
#endif

    bool success = true;
    success = success && parseHeader(f);
    success = success && validateFormat();
    success = success && parseMetadata(f);
    success = success && parseTensors(f);

    fclose(f);
    return success;
}

bool GGUFLoader::parseHeader(FILE* f) {
    size_t read = fread(&header.magic, sizeof(uint32_t), 1, f);
    if (read != 1) {
        LOG_ERROR("Failed to read GGUF magic");
        return false;
    }

    read = fread(&header.version, sizeof(uint32_t), 1, f);
    if (read != 1) {
        LOG_ERROR("Failed to read GGUF version");
        return false;
    }

    // GGUF v3 header uses 64-bit counts for tensors and metadata entries.
    read = fread(&header.tensorCount, sizeof(uint64_t), 1, f);
    if (read != 1) {
        LOG_ERROR("Failed to read tensor count");
        return false;
    }

    read = fread(&header.metadataKvCount, sizeof(uint64_t), 1, f);
    if (read != 1) {
        LOG_ERROR("Failed to read metadata KV count");
        return false;
    }

    LOG_INFO("GGUF Header: magic=0x" << std::hex << header.magic 
             << " version=" << std::dec << header.version
             << " tensorCount=" << header.tensorCount
             << " kvCount=" << header.metadataKvCount);

    return true;
}

bool GGUFLoader::validateFormat() {
    // GGUF magic: "GGUF" = 0x46554747
    const uint32_t GGUF_MAGIC = 0x46554747;
    
    if (header.magic != GGUF_MAGIC) {
        LOG_ERROR("Invalid GGUF magic number");
        return false;
    }

    if (header.version < 2) {
        LOG_ERROR("Unsupported GGUF version: " + std::to_string(header.version));
        return false;
    }

    return true;
}

std::string GGUFLoader::readString(FILE* f) {
    size_t read = 0;
    uint64_t len = 0;
    read = fread(&len, sizeof(uint64_t), 1, f);
    if (read != 1) {
        return "";
    }

    // If the length seems invalid, try a 32-bit length fallback (some GGUF variants)
    if(len == 0 || fileSize_ == 0 || len > (1ULL<<24) || static_cast<uint64_t>(ftell(f)) + len > fileSize_) {
        // rewind 8 bytes and try reading a 32-bit length
        if(fseek(f, -static_cast<long>(sizeof(uint64_t)), SEEK_CUR) != 0) {
            LOG_ERROR("Failed to rewind to try 32-bit length fallback");
            return "";
        }
        uint32_t len32 = 0;
        read = fread(&len32, sizeof(uint32_t), 1, f);
        if(read != 1) {
            LOG_ERROR("Failed to read 32-bit length fallback");
            return "";
        }
        len = len32;
        if(len == 0 || len > (1U<<24) || static_cast<uint64_t>(ftell(f)) + len > fileSize_) {
            LOG_ERROR("Invalid string length in GGUF after fallback: " << len);
            return "";
        }
    }

    long pos = ftell(f);
    if(pos < 0) { LOG_ERROR("ftell failed"); return ""; }
    if(static_cast<uint64_t>(pos) + len > fileSize_) {
        LOG_ERROR("String length exceeds file size in GGUF");
        return "";
    }

    std::vector<char> buffer(static_cast<size_t>(len));
    read = fread(buffer.data(), 1, len, f);
    if (read != (size_t)len) {
        LOG_ERROR("Failed to read string bytes from GGUF");
        return "";
    }

    return std::string(buffer.data(), len);
}

bool GGUFLoader::parseMetadata(FILE* f) {
    for (uint64_t i = 0; i < header.metadataKvCount; i++) {
        std::string key = readString(f);
        if (key.empty()) {
            LOG_ERROR("Failed to read metadata key");
            return false;
        }

        uint32_t valueType = 0;
        size_t read = fread(&valueType, sizeof(uint32_t), 1, f);
        if (read != 1) {
            LOG_ERROR("Failed to read metadata value type");
            return false;
        }

        std::string value;
        switch (valueType) {
            case 0: { // uint8
                uint8_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 1: { // int8
                int8_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 2: { // uint16
                uint16_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 3: { // int16
                int16_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 4: { // uint32
                uint32_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 5: { // int32
                int32_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 6: { // float32
                float v = 0.0f;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 7: { // bool
                uint8_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = v ? "true" : "false";
                break;
            }
            case 8: { // string
                value = readString(f);
                break;
            }
            case 9: { // array
                uint32_t arr_type = 0;
                uint64_t n_elems = 0;
                read = fread(&arr_type, sizeof(arr_type), 1, f);
                if (read != 1) return false;
                read = fread(&n_elems, sizeof(n_elems), 1, f);
                if (read != 1) return false;
                for (uint64_t j = 0; j < n_elems; ++j) {
                    switch (arr_type) {
                        case 0: { uint8_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 1: { int8_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 2: { uint16_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 3: { int16_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 4: { uint32_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 5: { int32_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 6: { float tmp = 0.0f; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 7: { uint8_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 8: { readString(f); break; }
                        case 10: { uint64_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 11: { int64_t tmp = 0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        case 12: { double tmp = 0.0; fread(&tmp, sizeof(tmp), 1, f); break; }
                        default: { value = "<unsupported-array>"; break; }
                    }
                }
                break;
            }
            case 10: { // uint64
                uint64_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 11: { // int64
                int64_t v = 0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            case 12: { // float64
                double v = 0.0;
                read = fread(&v, sizeof(v), 1, f);
                if (read != 1) return false;
                value = std::to_string(v);
                break;
            }
            default:
                LOG_WARN("Unknown metadata type: " + std::to_string(valueType));
                break;
        }

        metadata[key] = value;
    }

    return true;
}

bool GGUFLoader::parseTensors(FILE* f) {
    size_t read = 0;
    // Position after metadata is tensor data offset
    header.tensorDataOffset = ftell(f);

    LOG_INFO("GGUF contains " + std::to_string(header.tensorCount) + " tensors");

    for (uint64_t i = 0; i < header.tensorCount; i++) {
        GGUFTensor tensor;
        tensor.name = readString(f);

        if (tensor.name.empty()) {
            LOG_ERROR("Failed to read tensor name");
            return false;
        }

        read = fread(&tensor.ndim, sizeof(uint32_t), 1, f);
        if (read != 1) return false;

        tensor.shape.resize(tensor.ndim);
        for (uint32_t j = 0; j < tensor.ndim; j++) {
            uint64_t dim = 0;
            read = fread(&dim, sizeof(uint64_t), 1, f);
            if (read != 1) return false;
            tensor.shape[j] = dim;
        }

        read = fread(&tensor.type, sizeof(uint32_t), 1, f);
        if (read != 1) return false;

        read = fread(&tensor.offset, sizeof(uint64_t), 1, f);
        if (read != 1) return false;

        // Calculate size based on shape and type
        tensor.size = 1;
        for (auto dim : tensor.shape) {
            tensor.size *= dim;
        }

        // Account for quantization (e.g., Q4_0 = 4 bits per value)
        if (tensor.type == 2) { // Q4_0
            tensor.size = (tensor.size * 4) / 8;
        } else if (tensor.type == 0) { // F32
            tensor.size *= 4;
        } else if (tensor.type == 1) { // F16
            tensor.size *= 2;
        }

        tensors[tensor.name] = tensor;

        LOG_INFO("Tensor: " + tensor.name + " shape=" + std::to_string(tensor.shape[0]));
    }

    return true;
}

const GGUFTensor* GGUFLoader::getTensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string> GGUFLoader::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& pair : tensors) {
        names.push_back(pair.first);
    }
    return names;
}

bool GGUFLoader::streamTensorData(const std::string& tensorName, 
                                  void* buffer, size_t bufferSize) {
    const GGUFTensor* tensor = getTensor(tensorName);
    if (!tensor) {
        LOG_ERROR("Tensor not found: " + tensorName);
        return false;
    }

    if (bufferSize < tensor->size) {
        LOG_ERROR("Buffer too small for tensor: " + tensorName);
        return false;
    }

    FILE* f = fopen(filePath.c_str(), "rb");
    if (!f) {
        LOG_ERROR("Failed to open GGUF file for streaming");
        return false;
    }

    // Seek to tensor data
    fseek(f, header.tensorDataOffset + tensor->offset, SEEK_SET);
    
    size_t read = fread(buffer, 1, tensor->size, f);
    fclose(f);

    if (read != tensor->size) {
        LOG_ERROR("Failed to read complete tensor data");
        return false;
    }

    return true;
}

std::string GGUFLoader::getMetadata(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return "";
    }
    return it->second;
}

uint32_t GGUFLoader::getHiddenDim() const {
    std::string val = getMetadata("llama.embedding_length");
    if (val.empty()) val = getMetadata("embedding_length");
    return val.empty() ? 0 : std::stoul(val);
}

uint32_t GGUFLoader::getNumLayers() const {
    std::string val = getMetadata("llama.block_count");
    if (val.empty()) val = getMetadata("block_count");
    return val.empty() ? 0 : std::stoul(val);
}

uint32_t GGUFLoader::getNumHeads() const {
    std::string val = getMetadata("llama.attention.head_count");
    if (val.empty()) val = getMetadata("attention.head_count");
    return val.empty() ? 0 : std::stoul(val);
}

uint32_t GGUFLoader::getVocabSize() const {
    std::string val = getMetadata("llama.vocab_size");
    if (val.empty()) val = getMetadata("vocab_size");
    return val.empty() ? 0 : std::stoul(val);
}

uint64_t GGUFLoader::getTotalModelSize() const {
    uint64_t totalSize = 0;
    for (const auto& pair : tensors) {
        totalSize += pair.second.size;
    }
    return totalSize;
}

bool GGUFLoader::hasTensor(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}
