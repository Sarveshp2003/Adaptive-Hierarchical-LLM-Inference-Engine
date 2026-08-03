#pragma once

#include "runtime_contract.h"
#include "tensor_ops.h"
#include "../loader/mmap_weight_loader.h"

namespace runtime {

class CpuRuntime : public RuntimeContract {
public:
    bool initialize(const std::string &model_path) override;
    Tensor executeLayer(int layer_id, const Tensor &input) override;
    ModelMetadata metadata() const override;
    ~CpuRuntime() override;

private:
    ModelMetadata meta_;
    std::vector<float> weights_;
    // optional mmap weight loader kept alive for zero-copy views
    loader::MMapWeightLoader *mm_loader_ = nullptr;
};

}
