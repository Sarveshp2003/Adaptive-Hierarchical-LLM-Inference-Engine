#include "layer_loader.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include "custom_model_loader.h"
#include "../native-engine/include/GGUFLoader.h"

namespace {
bool isGGUFPath(const std::filesystem::path &path) {
    return path.has_extension() && path.extension().string().size() > 0 &&
           (path.extension().string().compare(0, 5, ".gguf") == 0 ||
            path.extension().string().compare(0, 5, ".GGUF") == 0);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int parseLayerIdFromTensorName(const std::string &name) {
    const std::string lowered = toLower(name);
    std::size_t pos = lowered.find("blk.");
    if (pos != std::string::npos) {
        const std::size_t start = pos + 4;
        std::size_t end = lowered.find('.', start);
        if (end == std::string::npos) end = lowered.size();
        const std::string number = lowered.substr(start, end - start);
        if (!number.empty()) {
            try { return std::stoi(number); } catch (...) {}
        }
    }

    const std::string marker = "layer";
    pos = lowered.find(marker);
    if (pos != std::string::npos) {
        const std::size_t start = pos + marker.size();
        std::size_t end = lowered.find('.', start);
        if (end == std::string::npos) end = lowered.size();
        const std::string number = lowered.substr(start, end - start);
        if (!number.empty()) {
            try { return std::stoi(number); } catch (...) {}
        }
    }
    return -1;
}

float halfToFloat(uint16_t value) {
    const uint32_t sign = (value & 0x8000u) << 16;
    const uint32_t exp = (value & 0x7C00u) >> 10;
    const uint32_t mant = value & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) return std::copysign(0.0f, static_cast<float>(sign));
        return std::ldexp(static_cast<float>(mant), -24) * (sign ? -1.0f : 1.0f);
    }
    if (exp == 0x1F) {
        return (mant == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(static_cast<float>(mant) + 1024.0f, exp - 25) * (sign ? -1.0f : 1.0f);
}

std::vector<float> readGGUFWeights(const std::string &model_path, int layer_id) {
    GGUFLoader loader;
    if (!loader.loadMetadata(model_path)) {
        throw std::runtime_error("LayerLoader: failed to parse GGUF metadata: " + model_path);
    }

    const auto tensor_names = loader.getTensorNames();
    std::string chosen_tensor;
    for (const auto &name : tensor_names) {
        const int parsed_id = parseLayerIdFromTensorName(name);
        if (parsed_id == layer_id) {
            chosen_tensor = name;
            break;
        }
    }
    if (chosen_tensor.empty() && !tensor_names.empty()) {
        for (const auto &name : tensor_names) {
            const std::string lower = toLower(name);
            if (lower.find("token_embd") != std::string::npos || lower.find("output") != std::string::npos) {
                chosen_tensor = name;
                break;
            }
        }
    }
    if (chosen_tensor.empty() && !tensor_names.empty()) {
        chosen_tensor = tensor_names.front();
    }

    const auto tensor = loader.getTensor(chosen_tensor);
    if (!tensor) {
        throw std::runtime_error("LayerLoader: GGUF tensor not found: " + chosen_tensor);
    }

    std::vector<uint8_t> raw(static_cast<size_t>(tensor->size));
    if (!loader.streamTensorData(chosen_tensor, raw.data(), raw.size())) {
        throw std::runtime_error("LayerLoader: failed to read GGUF tensor: " + chosen_tensor);
    }

    size_t bytes_per_element = 4;
    switch (tensor->type) {
        case 1: bytes_per_element = 2; break;
        case 2: bytes_per_element = 2; break;
        case 3: bytes_per_element = 1; break;
        case 4: bytes_per_element = 1; break;
        default: bytes_per_element = 4; break;
    }

    const size_t elems = std::max<size_t>(1, std::min<size_t>(8, tensor->size / std::max<size_t>(1, bytes_per_element)));
    std::vector<float> weights;
    weights.reserve(elems);
    for (size_t i = 0; i < elems; ++i) {
        if (tensor->type == 0) {
            float value{};
            std::memcpy(&value, raw.data() + i * bytes_per_element, sizeof(float));
            weights.push_back(value);
        } else if (tensor->type == 1) {
            uint16_t value{};
            std::memcpy(&value, raw.data() + i * bytes_per_element, sizeof(uint16_t));
            weights.push_back(halfToFloat(value));
        } else {
            weights.push_back(static_cast<float>(raw[i % raw.size()]));
        }
    }
    return weights;
}
}

LayerLoader::LayerLoader(const std::string &layers_dir) : dir_(layers_dir) {
    const std::filesystem::path path(layers_dir);
    if (isGGUFPath(path)) {
        model_path_ = path.string();
        is_gguf_model_ = true;
    }
}

loader::Tensor LayerLoader::load(int layer_id) {
    loader::Tensor t;
    if (is_gguf_model_) {
        const auto weights = readGGUFWeights(model_path_, layer_id);
        t.shape = { static_cast<int>(weights.size()) };
        t.data = weights;
        return t;
    }

    if (archiveExists()) {
        const auto archive = loadArchive();
        for (const auto &layer : archive.layers) {
            if (layer.layer_id == layer_id) {
                // fill loader::Tensor from archive weights
                t.shape = layer.shape;
                if (t.shape.empty()) t.shape = { static_cast<int>(layer.weights.size()) };
                t.data = layer.weights;
                return t;
            }
        }
        throw std::runtime_error("LayerLoader: layer not found in archive: " + std::to_string(layer_id));
    }

    std::string path = layerPath(dir_, layer_id);

    MmapLoader mm;
    if (!mm.mapFile(path)) {
        throw std::runtime_error("LayerLoader: failed to map file: " + path);
    }

    size_t bytes = mm.size();
    if (bytes % sizeof(float) != 0) {
        std::cerr << "LayerLoader: warning - layer file size not multiple of float32" << std::endl;
    }
    size_t n = bytes / sizeof(float);

    const float* src = static_cast<const float*>(mm.data());
    t.shape = { static_cast<int>(n) };
    t.data.resize(n);
    for (size_t i = 0; i < n; ++i) t.data[i] = src[i];

    return t;
}

bool LayerLoader::exists(int layer_id) const {
    if (is_gguf_model_) {
        try {
            (void)readGGUFWeights(model_path_, layer_id);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }
    if (archiveExists()) {
        try {
            const auto archive = loadArchive();
            for (const auto &layer : archive.layers) {
                if (layer.layer_id == layer_id) return true;
            }
        } catch (const std::exception &) {
            return false;
        }
    }
    std::string path = layerPath(dir_, layer_id);
    return std::filesystem::exists(path);
}

std::string LayerLoader::layerPath(const std::string &dir, int id) {
    return dir + "\\layer_" + std::to_string(id) + ".bin";
}

std::string LayerLoader::archivePath(const std::string &dir) {
    const std::vector<std::string> candidates = {
        dir + "\\model.adaptive",
        dir + "\\sample_model.adaptive"
    };
    for (const auto &candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

loader::ModelArchive LayerLoader::loadArchive() const {
    std::string path = archivePath(dir_);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("LayerLoader: archive not found: " + path);
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error("LayerLoader: failed to open archive: " + path);

    loader::ModelArchive arch;
    std::string line;
    // basic adaptive archive format (compatible with GGUP simple layout)
    if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading name");
    arch.name = line;
    if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading format");
    arch.format = line;
    if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading version");
    arch.version = std::stoi(line);

    if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading layer count");
    int n = std::stoi(line);
    for (int i = 0; i < n; ++i) {
        loader::LayerRecord lr;
        if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading layer id");
        lr.layer_id = std::stoi(line);
        if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading layer name");
        lr.name = line;
        if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading shape");
        // parse shape as comma or 'x' delimited
        lr.shape.clear();
        {
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, 'x')) {
                if (item.empty()) continue;
                std::stringstream inner(item);
                std::string sub;
                while (std::getline(inner, sub, ',')) {
                    if (!sub.empty()) {
                        lr.shape.push_back(std::stoi(sub));
                    }
                }
            }
        }
        if (!std::getline(in, line)) throw std::runtime_error("LayerLoader: unexpected EOF reading weights");
        lr.weights.clear();
        {
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) lr.weights.push_back(std::stof(item));
            }
        }
        arch.layers.push_back(lr);
    }
    return arch;
}

bool LayerLoader::archiveExists() const {
    return std::filesystem::exists(archivePath(dir_));
}

void LayerLoader::release(int /*layer_id*/) {
    // no-op in prototype
}
