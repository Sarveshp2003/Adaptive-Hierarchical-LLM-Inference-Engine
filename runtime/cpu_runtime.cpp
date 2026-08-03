#include "cpu_runtime.h"
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include "../util/logger.h"
#include "../loader/mmap_weight_loader.h"
#include "GGUFLoader.h"

namespace runtime {

bool CpuRuntime::initialize(const std::string &model_path) {
    std::filesystem::path p(model_path);
    std::filesystem::path weights_candidate = p.parent_path() / (p.stem().string() + ".weights");

    if (p.extension().string().compare(1, 4, "gguf") == 0) {
        GGUFLoader loader;
        if (loader.loadMetadata(model_path)) {
            meta_.path = model_path;
            meta_.format = "gguf";
            meta_.name = loader.getMetadata("general.name");
            if (meta_.name.empty()) {
                meta_.name = p.stem().string();
            }
            meta_.num_layers = static_cast<int64_t>(loader.getNumLayers());
            meta_.hidden_size = static_cast<int64_t>(loader.getHiddenDim());
            util::log_info("CPU runtime initialized from GGUF model: " + meta_.name + " layers=" + std::to_string(meta_.num_layers) + " hidden=" + std::to_string(meta_.hidden_size));
            return true;
        }
    }

    // Prefer mmap weights if companion .weights file exists

    if (std::filesystem::exists(weights_candidate)) {
        loader::MMapWeightLoader *wl = new loader::MMapWeightLoader();
        if (wl->mapFile(weights_candidate.string())) {
            // try index file
            std::filesystem::path idx = weights_candidate.string() + ".idx";
            if (std::filesystem::exists(idx)) {
                wl->loadIndex(idx.string());
                try {
                    // keep loader alive for zero-copy views
                    mm_loader_ = wl;
                    util::log_info("CPU runtime initialized using mmap weights (indexed, zero-copy enabled): " + weights_candidate.string());
                } catch (const std::exception &e) {
                    util::log_warn(std::string("CPU runtime failed reading indexed mmap weights: ") + e.what());
                    delete wl;
                    return false;
                }
            } else {
                try {
                    weights_ = wl->readFloatSlice(0, 4);
                    util::log_info("CPU runtime initialized using mmap weights: " + weights_candidate.string());
                    delete wl;
                } catch (const std::exception &e) {
                    util::log_warn(std::string("CPU runtime failed reading mmap weights: ") + e.what());
                    delete wl;
                    return false;
                }
            }

            meta_.path = model_path;
            meta_.format = "cpu-runtime-v1";
            meta_.name = "cpu-runtime";
            meta_.num_layers = 2;
            meta_.hidden_size = 4;
            return true;
        } else {
            delete wl;
        }
    }

    std::ifstream in(model_path, std::ios::binary);
    if (!in) {
        util::log_warn("CPU runtime failed to open model file: " + model_path);
        return false;
    }

    meta_.path = model_path;
    meta_.format = "cpu-runtime-v1";
    meta_.name = "cpu-runtime";
    meta_.num_layers = 2;
    meta_.hidden_size = 4;

    std::vector<float> weights = {0.5f, 0.25f, 0.1f, 0.05f};
    weights_ = std::move(weights);
    util::log_info("CPU runtime initialized from " + model_path);
    return true;
}

Tensor CpuRuntime::executeLayer(int layer_id, const Tensor &input) {
    if (input.data.empty()) {
        throw std::runtime_error("CPU runtime input is empty");
    }

    Tensor out;
    out.shape = input.shape;

    const size_t hidden = std::max<int64_t>(1, meta_.hidden_size > 0 ? meta_.hidden_size : static_cast<int64_t>(input.data.size()));
    std::vector<float> working = input.data;
    if (working.size() < hidden) {
        working.resize(hidden, 0.0f);
    }

    if (layer_id < 0) {
        out.data = working;
        return out;
    }

    std::vector<float> weights(hidden, 0.0f);
    for (size_t i = 0; i < weights.size(); ++i) {
        float base = 0.01f * static_cast<float>(i + 1 + layer_id);
        weights[i] = std::sin(base) * 0.1f + 0.05f;
    }

    for (size_t i = 0; i < working.size(); ++i) {
        working[i] = working[i] + weights[i % weights.size()];
    }

    if (layer_id % 2 == 0) {
        working = TensorOps::relu(working);
    } else {
        std::vector<float> mixed(working.size(), 0.0f);
        for (size_t i = 0; i < working.size(); ++i) {
            mixed[i] = working[i] * (0.5f + 0.01f * static_cast<float>(layer_id + 1));
        }
        working = std::move(mixed);
    }

    out.data = std::move(working);
    util::log_info("CPU runtime executed layer " + std::to_string(layer_id) + " for hidden=" + std::to_string(hidden));
    return out;
}

CpuRuntime::~CpuRuntime() {
    if (mm_loader_) {
        delete mm_loader_;
        mm_loader_ = nullptr;
    }
}

ModelMetadata CpuRuntime::metadata() const {
    return meta_;
}

}
