#pragma once

#include <string>
#include <cstddef>
#include <vector>
#include "weights_index.h"

class MmapLoader; // forward

namespace loader {

class MMapWeightLoader {
public:
    MMapWeightLoader();
    ~MMapWeightLoader();

    // Map a weight file into memory
    bool mapFile(const std::string &path);
    const void* data() const;
    size_t size() const;

    // Read a float array slice (copies into vector)
    std::vector<float> readFloatSlice(size_t offset_bytes, size_t float_count) const;

    // Load an index mapping and allow reading by layer id
    bool loadIndex(const std::string &idx_path);
    std::vector<float> readLayer(int layer_id) const;

    // Zero-copy view into mapped memory: returns pointer to float data and writes out float_count
    const float* readLayerView(int layer_id, size_t &out_float_count) const;

private:
    MmapLoader *mm_ = nullptr;
    std::vector<WeightIndexEntry> index_;
};

}
