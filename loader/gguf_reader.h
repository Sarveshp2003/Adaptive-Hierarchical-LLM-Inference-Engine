#pragma once

#include <string>
#include <vector>
#include "custom_model_loader.h"

namespace loader {

/**
 * Native GGUF reader using llama.cpp's gguf.h API.
 * Parses GGUF files and extracts tensor data into ModelArchive.
 */
class GGUFReader {
public:
    /**
     * Load a GGUF file and convert to ModelArchive.
     * Supports F16, F32, and quantized types (converts to float32).
     */
    static ModelArchive load(const std::string &path);

private:
    static float convertToFloat(const void *data, int ggml_type);
};

}  // namespace loader
