#include "runtime_adapter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "../util/logger.h"

namespace runtime {

bool RuntimeAdapter::initialize(const std::string &model_path) {
    cpu_backend_ = std::make_unique<CpuBackend>();
    gpu_backend_ = std::make_unique<GpuBackend>();

    // Respect environment override to force CPU backend
    const char *force_cpu = std::getenv("ADAPTIVELLM_FORCE_CPU");
    bool prefer_gpu = true;
    if (force_cpu && std::string(force_cpu) == "1") {
        prefer_gpu = false;
        util::log_info("ADAPTIVELLM_FORCE_CPU=1, forcing CPU backend");
    }

    if (prefer_gpu && gpu_backend_->load(model_path)) {
        active_backend_ = gpu_backend_.get();
        util::log_info("Runtime adapter initialized with GPU backend for " + model_path);
        return true;
    }

    if (cpu_backend_->load(model_path)) {
        active_backend_ = cpu_backend_.get();
        util::log_info("Runtime adapter initialized with CPU backend for " + model_path);
        return true;
    }

    return false;
}

Tensor RuntimeAdapter::executeLayer(int layer_id, const Tensor &input) {
    if (!active_backend_) {
        throw std::runtime_error("Runtime adapter is not initialized");
    }
    return active_backend_->runLayer(layer_id, input);
}

ModelMetadata RuntimeAdapter::metadata() const {
    if (!active_backend_) {
        return {};
    }
    return active_backend_->metadata();
}

std::vector<float> RuntimeAdapter::generateTokens(const std::vector<float> &prompt, size_t max_tokens, float temperature) {
    if (!active_backend_) {
        throw std::runtime_error("Runtime adapter is not initialized");
    }

    std::vector<float> current = prompt;
    std::vector<float> output;
    output.reserve(max_tokens);

    for (size_t i = 0; i < max_tokens; ++i) {
        Tensor input;
        input.data = current;
        input.shape.dims = {static_cast<int64_t>(current.size())};
        Tensor next = executeLayer(static_cast<int>(i % 2), input);
        float token = next.data.empty() ? 0.0f : next.data[0];
        if (temperature != 1.0f && temperature > 0.0f) {
            token = token / temperature;
        }
        output.push_back(token);
        current.assign(next.data.begin(), next.data.end());
    }

    return output;
}

}
