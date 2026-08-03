#pragma once

#include "runtime_contract.h"

namespace runtime {

class CpuBackend : public ModelBackend {
public:
    bool load(const std::string &path) override;
    Tensor runLayer(int layer_id, const Tensor &input) override;
    ModelMetadata metadata() const override;

private:
    ModelMetadata meta_;
};

}