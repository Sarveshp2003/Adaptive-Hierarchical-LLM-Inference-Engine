#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

/**
 * GGUF format support for loading quantized models.
 * 
 * GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

struct GGUFHeader {
    uint32_t magic;              // "GGUF" magic number
    uint32_t version;            // Format version
    uint64_t tensorCount;        // Number of tensors in the model
    uint64_t metadataKvCount;    // Number of metadata key-value pairs
    uint64_t tensorDataOffset;   // Offset to tensor data (set after metadata parsing)
};

struct GGUFTensor {
    std::string name;
    uint32_t ndim;
    std::vector<uint64_t> shape;
    uint32_t type;               // Data type (0=F32, 1=F16, 2=Q4_0, etc.)
    uint64_t offset;             // Offset in file
    size_t size;                 // Size in bytes
};

class GGUFLoader {
private:
    std::string filePath;
    GGUFHeader header;
    std::map<std::string, GGUFTensor> tensors;
    std::map<std::string, std::string> metadata;

    // File size for defensive parsing
    uint64_t fileSize_ = 0;

    /**
     * Parse GGUF file header
     */
    bool parseHeader(FILE* f);

    /**
     * Parse metadata key-value pairs
     */
    bool parseMetadata(FILE* f);

    /**
     * Parse tensor information
     */
    bool parseTensors(FILE* f);

    /**
     * Read string from file
     */
    std::string readString(FILE* f);

    /**
     * Validate GGUF magic and version
     */
    bool validateFormat();

public:
    GGUFLoader();
    ~GGUFLoader();

    /**
     * Load GGUF file metadata (non-blocking on tensor data)
     */
    bool loadMetadata(const std::string& path);

    /**
     * Get tensor by name
     */
    const GGUFTensor* getTensor(const std::string& name) const;

    /**
     * Get all tensor names
     */
    std::vector<std::string> getTensorNames() const;

    /**
     * Stream load tensor data from file into provided buffer
     */
    bool streamTensorData(const std::string& tensorName, void* buffer, size_t bufferSize);

    /**
     * Get metadata value
     */
    std::string getMetadata(const std::string& key) const;

    /**
     * Get model configuration from metadata
     */
    uint32_t getHiddenDim() const;
    uint32_t getNumLayers() const;
    uint32_t getNumHeads() const;
    uint32_t getVocabSize() const;

    /**
     * Get total model size in bytes
     */
    uint64_t getTotalModelSize() const;

    /**
     * Check if tensor exists
     */
    bool hasTensor(const std::string& name) const;
};

#endif
