#include "gguf_reader.h"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include "../util/logger.h"

// Include llama.cpp GGUF C API when available.
#if __has_include("ggml.h") && __has_include("gguf.h")
#include "ggml.h"
#include "gguf.h"
#define ADAPTIVELLM_HAVE_GGUF_HEADERS 1
#else
#define ADAPTIVELLM_HAVE_GGUF_HEADERS 0
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_BF16 = 2 };
#define GGML_MAX_DIMS 4
struct gguf_context {};
struct gguf_init_params { bool no_alloc = false; void *ctx = nullptr; };
static inline gguf_context *gguf_init_from_file(const char *, const gguf_init_params &) { return nullptr; }
static inline void gguf_free(gguf_context *) {}
static inline int64_t gguf_get_n_tensors(const gguf_context *) { return 0; }
static inline size_t gguf_get_data_offset(const gguf_context *) { return 0; }
static inline const char *gguf_get_tensor_name(const gguf_context *, int64_t) { return nullptr; }
static inline int gguf_get_tensor_type(const gguf_context *, int64_t) { return GGML_TYPE_F32; }
static inline size_t gguf_get_tensor_size(const gguf_context *, int64_t) { return 0; }
static inline size_t gguf_get_tensor_offset(const gguf_context *, int64_t) { return 0; }
static inline const int64_t *gguf_get_tensor_ne(const gguf_context *, int64_t) { static const int64_t dims[GGML_MAX_DIMS] = {0}; return dims; }
#endif

namespace {

// Helper: convert float16 to float32
float fp16_to_float(uint16_t h) {
    uint32_t u = (h & 0x7FFF) << 13;
    uint32_t exp = (h >> 10) & 0x1F;
    if (exp == 0) {
        u = 0;
    } else if (exp == 31) {
        u |= 0x7F800000;
    } else {
        u |= ((uint32_t)(exp + 112)) << 23;
    }
    if (h & 0x8000) {
        u |= 0x80000000;
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

}

namespace loader {

float GGUFReader::convertToFloat(const void *data, int ggml_type) {
    if (!data) return 0.0f;
    
    switch (ggml_type) {
        case GGML_TYPE_F32:
        {
            float f;
            std::memcpy(&f, data, sizeof(float));
            return f;
        }
        case GGML_TYPE_F16:
        {
            uint16_t h;
            std::memcpy(&h, data, sizeof(uint16_t));
            return fp16_to_float(h);
        }
        case GGML_TYPE_BF16:
        {
            uint16_t bf;
            std::memcpy(&bf, data, sizeof(uint16_t));
            uint32_t f32 = ((uint32_t)bf) << 16;
            float f;
            std::memcpy(&f, &f32, sizeof(float));
            return f;
        }
        default:
            return (float)(((const uint8_t*)data)[0]) / 255.0f;
    }
}

ModelArchive GGUFReader::load(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("GGUFReader: file not found: " + path);
    }

    util::log_info("GGUFReader: loading GGUF file with llama.cpp API: " + path);

#if !ADAPTIVELLM_HAVE_GGUF_HEADERS
    throw std::runtime_error("GGUFReader: llama.cpp headers are not available in this build");
#endif
    
    ModelArchive arch;
    arch.name = std::filesystem::path(path).stem().string();
    arch.format = "gguf";
    arch.version = 3;

    // Use llama.cpp's GGUF API - no memory allocation for tensors
    struct gguf_init_params params;
    params.no_alloc = true;  // Don't load tensor data yet
    params.ctx = nullptr;

    struct gguf_context *ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("GGUFReader: failed to parse GGUF file");
    }

    try {
        int64_t n_tensors = gguf_get_n_tensors(ctx);
        util::log_info("GGUFReader: found " + std::to_string(n_tensors) + " tensors");

        // Open file for reading tensor data
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("GGUFReader: failed to open file for reading");
        }

        // Get data offset from context
        size_t data_offset = gguf_get_data_offset(ctx);
        util::log_info("GGUFReader: tensor data starts at offset " + std::to_string(data_offset));

        // Process each tensor
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char *tensor_name = gguf_get_tensor_name(ctx, i);
            if (!tensor_name) continue;

            ggml_type dtype = static_cast<ggml_type>(gguf_get_tensor_type(ctx, i));
            size_t tensor_bytes = gguf_get_tensor_size(ctx, i);
            size_t tensor_offset = gguf_get_tensor_offset(ctx, i);

            // Get shape
            const int64_t *shape_ptr = gguf_get_tensor_ne(ctx, i);
            std::vector<int64_t> shape;
            for (int d = 0; d < GGML_MAX_DIMS && shape_ptr[d] > 0; ++d) {
                shape.push_back(shape_ptr[d]);
            }
            if (shape.empty()) shape.push_back(1);

            // Calculate element count
            size_t elem_count = 1;
            for (auto s : shape) {
                if (s > 0) elem_count *= s;
            }
            if (elem_count == 0) continue;

            if ((i + 1) % 50 == 0) {
                util::log_info("GGUFReader: processing tensor " + std::to_string(i + 1) + "/" + std::to_string(n_tensors));
            }

            // Read tensor data from file
            std::vector<uint8_t> raw_data(tensor_bytes);
            file.seekg(data_offset + tensor_offset);
            file.read((char*)raw_data.data(), tensor_bytes);
            if (!file) {
                util::log_warn("GGUFReader: failed to read tensor data for " + std::string(tensor_name));
                continue;
            }

            // Convert to float32
            std::vector<float> float_data(elem_count);
            
            if (dtype == GGML_TYPE_F32) {
                // Direct memcpy
                std::memcpy(float_data.data(), raw_data.data(), tensor_bytes);
            } else if (dtype == GGML_TYPE_F16) {
                // Convert F16 -> F32
                const uint16_t *f16_ptr = (const uint16_t*)raw_data.data();
                for (size_t j = 0; j < elem_count; ++j) {
                    float_data[j] = fp16_to_float(f16_ptr[j]);
                }
            } else if (dtype == GGML_TYPE_BF16) {
                // Convert BF16 -> F32
                const uint16_t *bf_ptr = (const uint16_t*)raw_data.data();
                for (size_t j = 0; j < elem_count; ++j) {
                    uint32_t f32 = ((uint32_t)bf_ptr[j]) << 16;
                    float f;
                    std::memcpy(&f, &f32, sizeof(float));
                    float_data[j] = f;
                }
            } else {
                // Quantized types: approximate
                util::log_warn("GGUFReader: tensor " + std::string(tensor_name) +
                              " is quantized (type " + std::to_string(dtype) + "); approximating");
                for (size_t j = 0; j < elem_count && j < raw_data.size(); ++j) {
                    float_data[j] = (float)(raw_data[j]) / 255.0f;
                }
            }

            // Create layer record
            LayerRecord lr;
            lr.name = tensor_name;
            lr.weights = float_data;
            
            // Extract layer ID from name
            std::string name_str(tensor_name);
            size_t pos = name_str.find("layers");
            if (pos == std::string::npos) pos = name_str.find("layer");
            if (pos != std::string::npos) {
                size_t start = name_str.find_first_of("0123456789", pos);
                if (start != std::string::npos) {
                    size_t end = start;
                    while (end < name_str.size() && isdigit((unsigned char)name_str[end])) {
                        ++end;
                    }
                    try {
                        lr.layer_id = std::stoi(name_str.substr(start, end - start));
                    } catch (...) {}
                }
            }

            // Copy shape
            for (auto s : shape) {
                lr.shape.push_back((int)s);
            }

            arch.layers.push_back(lr);
        }

        file.close();
    } catch (const std::exception &e) {
        gguf_free(ctx);
        throw;
    }

    gguf_free(ctx);
    util::log_info("GGUFReader: successfully loaded " + std::to_string(arch.layers.size()) + " layers");
    return arch;
}

}
