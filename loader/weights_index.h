#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace loader {

struct WeightIndexEntry {
    int layer_id;
    size_t offset_bytes;
    size_t float_count;
};

class WeightsIndex {
public:
    // Load index from a simple text format: one entry per line: <layer_id> <offset_bytes> <float_count>
    static std::vector<WeightIndexEntry> load(const std::string &path);
};

}
