#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace runtime {

struct TensorShape {
    std::vector<int64_t> dims;
};

struct Tensor {
    std::vector<float> data;
    TensorShape shape;
};

struct ModelMetadata {
    std::string name;
    std::string format;
    std::string path;
    int64_t num_layers = 0;
    int64_t hidden_size = 0;
};

class ModelBackend {
public:
    virtual ~ModelBackend() = default;
    virtual bool load(const std::string &path) = 0;
    virtual Tensor runLayer(int layer_id, const Tensor &input) = 0;
    virtual ModelMetadata metadata() const = 0;
};

class RuntimeContract {
public:
    virtual ~RuntimeContract() = default;
    virtual bool initialize(const std::string &model_path) = 0;
    virtual Tensor executeLayer(int layer_id, const Tensor &input) = 0;
    virtual ModelMetadata metadata() const = 0;
};

}