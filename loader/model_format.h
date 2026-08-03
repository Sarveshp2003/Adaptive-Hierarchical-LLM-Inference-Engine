#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace loader {

struct LayerRecord {
    int layer_id = 0;
    std::string name;
    std::vector<int> shape;
    std::vector<float> weights;
};

struct ModelArchive {
    std::string name;
    std::string format = "adaptive-custom-v1";
    int version = 1;
    std::vector<LayerRecord> layers;
};

}