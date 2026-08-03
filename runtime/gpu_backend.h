#pragma once

#include "runtime_contract.h"
#include "../loader/mmap_weight_loader.h"
#include "../loader/weights_index.h"

namespace runtime {

class GpuBackend : public ModelBackend {
public:
    explicit GpuBackend(int device_id = 0);

    bool load(const std::string &path) override;
    Tensor runLayer(int layer_id, const Tensor &input) override;
    ModelMetadata metadata() const override;

    bool available() const;
    int deviceId() const;

private:
    ModelMetadata meta_;
    int device_id_ = 0;
    bool available_ = false;

    // MMap weight loader and index (for adaptive archives)
    loader::MMapWeightLoader *weight_loader_ = nullptr;
    std::vector<loader::WeightIndexEntry> weight_index_entries_;

    struct ParsedTensor {
        std::string name;
        std::vector<int> shape;
        size_t index_pos; // position in index file
    };
    std::vector<ParsedTensor> parsed_tensors_;
};

}
