#include "gguf_adapter.h"
#include <filesystem>
#include <stdexcept>
#include "ggup_loader.h"
#include "gguf_reader.h"
#include "../util/logger.h"

#if defined(__has_include)
# if __has_include(<safetensors.hh>)
#  include <safetensors.hh>
#  define ADAPTIVELLM_HAVE_SAFETENSORS 1
# endif
#endif

namespace loader {

ModelArchive GGUFAdapter::load(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("GGUFAdapter: file not found: " + path);
    }

    // Try native GGUF reader first (most reliable for GGUF files)
    try {
        util::log_info("GGUFAdapter: trying native GGUF reader for " + path);
        return GGUFReader::load(path);
    } catch (const std::exception &e) {
        util::log_warn(std::string("GGUFAdapter: native GGUF reader failed: ") + e.what());
    }

#ifdef ADAPTIVELLM_HAVE_SAFETENSORS
    util::log_info("GGUFAdapter: trying safetensors reader for " + path);
    ModelArchive arch;
    arch.name = "gguf-model";
    arch.format = "gguf";
    arch.version = 1;

    // Warning: the safetensors-cpp API surface may differ between versions.
    // The following is a best-effort integration using common API patterns.
    try {
        safetensors::safetensors_t st;
        std::string warn, err;
        if (!safetensors::load_from_file(path, &st, &warn, &err)) {
            throw std::runtime_error(std::string("safetensors load failed: ") + err);
        }

        const auto &names = st.tensors.keys();
        for (const auto &name : names) {
            safetensors::tensor_t info;
            if (!st.tensors.at(name, &info)) continue;

            loader::LayerRecord lr;
            lr.name = name;
            // heuristic layer id
            size_t pos = name.find("layer");
            if (pos != std::string::npos) {
                size_t start = name.find_first_of("0123456789", pos);
                if (start != std::string::npos) {
                    size_t end = start;
                    while (end < name.size() && isdigit((unsigned char)name[end])) ++end;
                    lr.layer_id = std::stoi(name.substr(start, end - start));
                }
            }

            // copy shape
            lr.shape.clear();
            for (size_t s : info.shape) lr.shape.push_back((int)s);

            // read data buffer pointer
            const uint8_t *buf = nullptr;
            if (st.mmaped && st.databuffer_addr) buf = st.databuffer_addr;
            else if (!st.storage.empty()) buf = st.storage.data();
            else buf = nullptr;
            if (!buf) continue;

            size_t begin = info.data_offsets[0];
            size_t end = info.data_offsets[1];
            size_t bytes = end - begin;
            size_t count = safetensors::get_shape_size(info);

            // support float32 and float16 (FP16)
            if (info.dtype == safetensors::kFLOAT32) {
                const float *fptr = reinterpret_cast<const float*>(buf + begin);
                lr.weights.resize(count);
                memcpy(lr.weights.data(), fptr, count * sizeof(float));
            } else if (info.dtype == safetensors::kFLOAT16 || info.dtype == safetensors::kBFLOAT16) {
                const uint16_t *sptr = reinterpret_cast<const uint16_t*>(buf + begin);
                lr.weights.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    if (info.dtype == safetensors::kFLOAT16)
                        lr.weights[i] = safetensors::fp16_to_float(sptr[i]);
                    else
                        lr.weights[i] = safetensors::bfloat16_to_float(sptr[i]);
                }
            } else {
                // unsupported dtype for now
                continue;
            }

            arch.layers.push_back(lr);
        }
        return arch;
    } catch (const std::exception &e) {
        util::log_warn(std::string("GGUFAdapter: safetensors reader failed: ") + e.what());
    }
#endif

    util::log_info("GGUFAdapter: falling back to GGUPLoader text format for " + path);
    return GGUPLoader::load(path);
}

}