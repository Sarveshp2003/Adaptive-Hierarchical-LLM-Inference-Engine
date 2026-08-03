#include "ggup_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace loader {

namespace {

std::string trim(const std::string &s) {
    size_t a = 0;
    while (a < s.size() && isspace((unsigned char)s[a])) ++a;
    size_t b = s.size();
    while (b > a && isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

std::vector<int> parseShape(const std::string &line) {
    std::vector<int> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, 'x')) {
        if (!item.empty()) out.push_back(std::stoi(trim(item)));
    }
    return out;
}

std::vector<float> parseWeights(const std::string &line) {
    std::vector<float> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(trim(item)));
    }
    return out;
}

}

ModelArchive GGUPLoader::load(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("GGUPLoader: file not found: " + path);
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error("GGUPLoader: failed to open: " + path);

    ModelArchive arch;
    std::string line;
    std::getline(in, line); arch.name = trim(line);
    std::getline(in, line); arch.format = trim(line);
    std::getline(in, line); arch.version = std::stoi(trim(line));

    // number of layers
    std::getline(in, line);
    int n = std::stoi(trim(line));
    for (int i = 0; i < n; ++i) {
        LayerRecord lr;
        std::getline(in, line); lr.layer_id = std::stoi(trim(line));
        std::getline(in, line); lr.name = trim(line);
        std::getline(in, line); lr.shape = parseShape(trim(line));
        std::getline(in, line); lr.weights = parseWeights(trim(line));
        arch.layers.push_back(lr);
    }
    return arch;
}

}