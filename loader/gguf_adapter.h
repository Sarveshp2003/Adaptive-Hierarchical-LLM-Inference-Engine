#pragma once

#include "model_format.h"
#include <string>

namespace loader {

// Adapter that attempts to use a real GGUF/safetensors reader when available.
// Falls back to the lightweight GGUPLoader if no external reader is present.
class GGUFAdapter {
public:
    static ModelArchive load(const std::string &path);
};

}