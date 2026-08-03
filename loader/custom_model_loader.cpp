#include "custom_model_loader.h"
#include "ggup_loader.h"
#include "gguf_adapter.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace loader {
namespace {

std::string readLine(std::ifstream &in) {
    std::string line;
    std::getline(in, line);
    return line;
}

int parseInt(const std::string &value) {
    return std::stoi(value);
}

std::vector<int> parseShape(const std::string &value) {
    std::vector<int> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

std::vector<float> parseWeights(const std::string &value) {
    std::vector<float> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
    return out;
}

} // namespace

ModelArchive CustomModelLoader::load(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("CustomModelLoader: file not found: " + path);
    }

    // Detect GGUP (.ggup) or GGUF (.gguf) and delegate
    std::filesystem::path p(path);
    auto ext = p.extension().string();
    for (auto &c : ext) c = (char)tolower(c);
    if (ext == ".ggup") {
        return GGUPLoader::load(path);
    }
    if (ext == ".gguf") {
        return GGUFAdapter::load(path);
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("CustomModelLoader: failed to open: " + path);
    }

    ModelArchive archive;
    archive.name = readLine(in);
    archive.format = readLine(in);
    archive.version = parseInt(readLine(in));

    int layer_count = parseInt(readLine(in));
    for (int i = 0; i < layer_count; ++i) {
        LayerRecord layer;
        layer.layer_id = parseInt(readLine(in));
        layer.name = readLine(in);
        layer.shape = parseShape(readLine(in));
        layer.weights = parseWeights(readLine(in));
        archive.layers.push_back(layer);
    }

    return archive;
}

}