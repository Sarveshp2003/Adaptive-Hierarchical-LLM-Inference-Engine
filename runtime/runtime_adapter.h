#pragma once

#include <memory>
#include "runtime_contract.h"
#include "cpu_backend.h"
#include "gpu_backend.h"

namespace runtime {

class RuntimeAdapter : public RuntimeContract {
public:
    bool initialize(const std::string &model_path) override;
    Tensor executeLayer(int layer_id, const Tensor &input) override;
    ModelMetadata metadata() const override;

    std::vector<float> generateTokens(const std::vector<float> &prompt, size_t max_tokens = 4, float temperature = 1.0f);

private:
    std::unique_ptr<ModelBackend> cpu_backend_;
    std::unique_ptr<ModelBackend> gpu_backend_;
    ModelBackend *active_backend_ = nullptr;
};

}
