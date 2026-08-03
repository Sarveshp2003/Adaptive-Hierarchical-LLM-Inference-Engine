#include "mmap_weight_loader.h"
#include "mmap_loader.h"
#include "weights_index.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace loader {

MMapWeightLoader::MMapWeightLoader() : mm_(nullptr) {}

MMapWeightLoader::~MMapWeightLoader() {
    if (mm_) {
        mm_->unmap();
        delete mm_;
        mm_ = nullptr;
    }
}

bool MMapWeightLoader::mapFile(const std::string &path) {
    if (mm_) { mm_->unmap(); delete mm_; mm_ = nullptr; }
    mm_ = new MmapLoader();
    if (!mm_->mapFile(path)) {
        delete mm_;
        mm_ = nullptr;
        return false;
    }
    return true;
}

const void* MMapWeightLoader::data() const {
    if (!mm_) return nullptr;
    return mm_->data();
}

size_t MMapWeightLoader::size() const {
    if (!mm_) return 0;
    return mm_->size();
}

std::vector<float> MMapWeightLoader::readFloatSlice(size_t offset_bytes, size_t float_count) const {
    if (!mm_) throw std::runtime_error("MMapWeightLoader: not mapped");
    size_t bytes = float_count * sizeof(float);
    if (offset_bytes + bytes > mm_->size()) throw std::out_of_range("readFloatSlice out of range");
    const float *src = reinterpret_cast<const float*>(static_cast<const char*>(mm_->data()) + offset_bytes);
    std::vector<float> out(float_count);
    std::memcpy(out.data(), src, bytes);
    return out;
}

bool MMapWeightLoader::loadIndex(const std::string &idx_path) {
    try {
        index_ = loader::WeightsIndex::load(idx_path);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<float> MMapWeightLoader::readLayer(int layer_id) const {
    for (const auto &e : index_) {
        if (e.layer_id == layer_id) {
            return readFloatSlice(e.offset_bytes, e.float_count);
        }
    }
    throw std::runtime_error("MMapWeightLoader: layer id not found in index");
}

const float* MMapWeightLoader::readLayerView(int layer_id, size_t &out_float_count) const {
    if (!mm_) throw std::runtime_error("MMapWeightLoader: not mapped");
    for (const auto &e : index_) {
        if (e.layer_id == layer_id) {
            if (e.offset_bytes + e.float_count * sizeof(float) > mm_->size()) throw std::out_of_range("readLayerView out of range");
            out_float_count = e.float_count;
            const float *ptr = reinterpret_cast<const float*>(static_cast<const char*>(mm_->data()) + e.offset_bytes);
            return ptr;
        }
    }
    throw std::runtime_error("MMapWeightLoader: layer id not found in index");
}

}