#pragma once

#include "model_format.h"
#include <string>

namespace loader {

class GGUPLoader {
public:
    // Load a minimal GGUP-style archive (placeholder parser for prototype)
    static ModelArchive load(const std::string &path);
};

}