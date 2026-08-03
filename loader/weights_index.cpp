#include "weights_index.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace loader {

std::vector<WeightIndexEntry> WeightsIndex::load(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("WeightsIndex: failed to open index: " + path);
    std::vector<WeightIndexEntry> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        WeightIndexEntry e;
        ss >> e.layer_id >> e.offset_bytes >> e.float_count;
        if (!ss) throw std::runtime_error("WeightsIndex: parse error in " + path);
        out.push_back(e);
    }
    return out;
}

}