#include "cpu_backend.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "../util/logger.h"
#include "GGUFLoader.h"

namespace runtime {

namespace {

bool loadGGUFMetadata(const std::string &path, ModelMetadata &meta) {
    if (!std::filesystem::exists(path)) {
        return false;
    }

    std::filesystem::path p(path);
    if (p.extension().string().size() < 2 || p.extension().string().compare(1, 4, "gguf") != 0) {
        return false;
    }

    GGUFLoader loader;
    if (!loader.loadMetadata(path)) {
        util::log_warn("Failed to parse GGUF metadata from " + path);
        return false;
    }

    meta.path = path;
    meta.format = "gguf";
    meta.name = loader.getMetadata("general.name");
    if (meta.name.empty()) {
        meta.name = p.stem().string();
    }
    meta.num_layers = static_cast<int64_t>(loader.getNumLayers());
    meta.hidden_size = static_cast<int64_t>(loader.getHiddenDim());

    util::log_info("Loaded GGUF metadata from " + path + " layers=" + std::to_string(meta.num_layers) + " hidden=" + std::to_string(meta.hidden_size));
    return true;
}

} // namespace

bool CpuBackend::load(const std::string &path) {
    if (loadGGUFMetadata(path, meta_)) {
        return true;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        util::log_warn("Failed to load model file: " + path);
        return false;
    }
    meta_.path = path;
    meta_.format = "custom-binary";
    meta_.name = "cpu-prototype";
    meta_.num_layers = 1;
    meta_.hidden_size = 4;
    util::log_info("Loaded CPU backend model from " + path);
    return true;
}

Tensor CpuBackend::runLayer(int layer_id, const Tensor &input) {
    (void)layer_id;
    Tensor out = input;
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = out.data[i] * 1.01f;
    }
    util::log_info("Executed CPU backend layer " + std::to_string(layer_id));
    return out;
}

ModelMetadata CpuBackend::metadata() const {
    return meta_;
}

}