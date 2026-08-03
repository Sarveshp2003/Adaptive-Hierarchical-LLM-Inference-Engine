#pragma once

#include <string>
#include "model_format.h"

namespace loader {

class CustomModelLoader {
public:
    static ModelArchive load(const std::string &path);
};

}