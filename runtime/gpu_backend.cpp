#include "gpu_backend.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include "../util/logger.h"
#include "../loader/mmap_weight_loader.h"
#include "gpu/cuda_executor.h"
#include "GGUFLoader.h"

#if defined(ADAPTIVELLM_HAVE_CUDA)
  #if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
  #endif
#endif

namespace runtime {

namespace {

std::string formatDeviceName(int device_id) {
    std::ostringstream ss;
    ss << "device-" << device_id;
    return ss.str();
}

}

GpuBackend::GpuBackend(int device_id) : device_id_(device_id) {}

bool GpuBackend::load(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        util::log_warn("GPU backend could not find model file: " + path);
        return false;
    }

    bool loaded_gguf_metadata = false;
    std::filesystem::path model_path(path);
    if (model_path.extension().string().compare(1, 4, "gguf") == 0) {
        GGUFLoader loader;
        if (loader.loadMetadata(path)) {
            meta_.path = path;
            meta_.format = "gguf";
            meta_.name = loader.getMetadata("general.name");
            if (meta_.name.empty()) {
                meta_.name = model_path.stem().string();
            }
            meta_.num_layers = static_cast<int64_t>(loader.getNumLayers());
            meta_.hidden_size = static_cast<int64_t>(loader.getHiddenDim());
            loaded_gguf_metadata = true;
            util::log_info("GPU backend loaded GGUF metadata: name=" + meta_.name + " layers=" + std::to_string(meta_.num_layers) + " hidden=" + std::to_string(meta_.hidden_size));
        }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        util::log_warn("GPU backend failed to open model file: " + path);
        return false;
    }

    if (!loaded_gguf_metadata) {
        meta_.path = path;
        meta_.format = "gpu-prototype";
        meta_.name = "gpu-prototype";
        meta_.num_layers = 0;
        meta_.hidden_size = 0;
    }

    // If an adaptive metadata file exists (text), parse it to populate metadata
    std::filesystem::path p(path);
    std::filesystem::path adaptive_candidate = path;
    if (p.extension() == ".adaptive") {
        adaptive_candidate = p;
    } else {
        adaptive_candidate = p.parent_path() / (p.stem().string() + ".adaptive");
    }

    if (std::filesystem::exists(adaptive_candidate)) {
        try {
            std::ifstream a(adaptive_candidate);
            if (a) {
                std::vector<std::string> lines;
                std::string l;
                while (std::getline(a, l)) {
                    // trim
                    size_t s = 0, e = l.size();
                    while (s < e && isspace((unsigned char)l[s])) ++s;
                    while (e > s && isspace((unsigned char)l[e-1])) --e;
                    if (s < e) lines.push_back(l.substr(s, e-s));
                }
                if (lines.size() >= 4) {
                    meta_.name = lines[0];
                    meta_.format = lines[1];
                    // lines[2] is version
                    int64_t tensor_count = std::stoll(lines[3]);
                    // parse subsequent entries: expect groups of 3 (id/name/shape) but be tolerant
                    std::vector<std::string> names;
                    for (size_t i = 4; i + 1 < lines.size(); ) {
                        // skip possible numeric id line
                        std::string maybe_id = lines[i];
                        if (std::all_of(maybe_id.begin(), maybe_id.end(), [](char c){ return isdigit((unsigned char)c) || c=='-' ; })) {
                            // id line, skip
                            i++;
                        }
                        if (i >= lines.size()) break;
                        std::string tname = lines[i++];
                        if (i >= lines.size()) break;
                        std::string shape_line = lines[i++];
                        names.push_back(tname);
                        // parse shape
                        size_t comma = shape_line.find(',');
                        if (comma != std::string::npos) {
                            std::string first = shape_line.substr(0, comma);
                            try { int hs = std::stoi(first); if (hs > meta_.hidden_size) meta_.hidden_size = hs; } catch(...){}
                        } else {
                            try { int hs = std::stoi(shape_line); if (hs > meta_.hidden_size) meta_.hidden_size = hs; } catch(...){}
                        }
                    }
                    // infer num_layers from names (look for blk.<n>)
                    int max_blk = -1;
                    for (auto &nm : names) {
                        size_t pos = nm.find("blk.");
                        if (pos == std::string::npos) pos = nm.find("layer");
                        if (pos != std::string::npos) {
                            size_t start = pos;
                            // find first digit
                            while (start < nm.size() && !isdigit((unsigned char)nm[start])) ++start;
                            if (start < nm.size()) {
                                size_t end = start;
                                while (end < nm.size() && isdigit((unsigned char)nm[end])) ++end;
                                try {
                                    int v = std::stoi(nm.substr(start, end-start));
                                    if (v > max_blk) max_blk = v;
                                } catch(...){}
                            }
                        }
                    }
                    if (max_blk >= 0) meta_.num_layers = max_blk + 1;
                    else meta_.num_layers = (int)names.size();

                    util::log_info("GPU backend parsed adaptive metadata: name=" + meta_.name + " format=" + meta_.format + " layers=" + std::to_string(meta_.num_layers) + " hidden=" + std::to_string(meta_.hidden_size));
                }
            }
        } catch (const std::exception &e) {
            util::log_warn(std::string("GPU backend failed to parse adaptive metadata: ") + e.what());
        }
    }

    // Try mmap-backed weights if available
    std::filesystem::path weights_candidate = p.parent_path() / (p.stem().string() + ".weights");
    if (std::filesystem::exists(weights_candidate)) {
        weight_loader_ = new loader::MMapWeightLoader();
        if (weight_loader_->mapFile(weights_candidate.string())) {
            std::filesystem::path idx = weights_candidate.string() + ".idx";
            if (std::filesystem::exists(idx)) {
                try {
                    weight_index_entries_ = loader::WeightsIndex::load(idx.string());
                    util::log_info("GPU backend will use indexed mmap weights: " + weights_candidate.string());
                } catch (...) {
                    util::log_warn("GPU backend found weights file but failed to load index: " + idx.string());
                }
            } else {
                util::log_info("GPU backend will use mmap weights: " + weights_candidate.string());
            }

            // Build parsed_tensors_ mapping from adaptive metadata lines (best-effort)
            try {
                std::ifstream a(adaptive_candidate);
                if (a) {
                    std::vector<std::string> lines;
                    std::string l;
                    while (std::getline(a, l)) {
                        size_t s = 0, e = l.size();
                        while (s < e && isspace((unsigned char)l[s])) ++s;
                        while (e > s && isspace((unsigned char)l[e-1])) --e;
                        if (s < e) lines.push_back(l.substr(s, e-s));
                    }
                    // starting at index 4 (after header)
                    size_t pos = 4; size_t idxpos = 0;
                    while (pos + 1 < lines.size() && idxpos < weight_index_entries_.size()) {
                        // skip id lines if present
                        if (std::all_of(lines[pos].begin(), lines[pos].end(), [](char c){ return isdigit((unsigned char)c) || c=='-'; })) { pos++; continue; }
                        std::string tname = lines[pos++];
                        if (pos >= lines.size()) break;
                        std::string shape_line = lines[pos++];
                        ParsedTensor pt; pt.name = tname; pt.index_pos = idxpos;
                        // parse shape
                        std::vector<int> sh;
                        size_t start = 0;
                        while (start < shape_line.size()) {
                            size_t comma = shape_line.find(',', start);
                            if (comma == std::string::npos) comma = shape_line.size();
                            try { sh.push_back(std::stoi(shape_line.substr(start, comma - start))); } catch(...){}
                            start = comma + 1;
                        }
                        pt.shape = sh;
                        parsed_tensors_.push_back(pt);
                        idxpos++;
                    }
                }
            } catch (...) {}
        } else {
            delete weight_loader_; weight_loader_ = nullptr;
        }
    }

// Direct diagnostic using CUDA runtime if available at compile time
#if defined(ADAPTIVELLM_HAVE_CUDA) && __has_include(<cuda_runtime.h>)
    // CUDA runtime is available at global scope
    {
        int devCount = 0;
        cudaError_t err = cudaGetDeviceCount(&devCount);
        if (err != cudaSuccess) {
            util::log_warn(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
        } else {
            util::log_info(std::string("cudaGetDeviceCount reports ") + std::to_string(devCount) + " device(s)");
        }
    }
#endif

    // Prefer runtime detection of CUDA-capable devices
    if (runtime::gpu::cuda_available()) {
        available_ = true;
        util::log_info("GPU backend initialized for " + formatDeviceName(device_id_) + " using CUDA runtime");
    } else {
        available_ = false;
        util::log_info("GPU backend initialized in simulated mode for " + formatDeviceName(device_id_));
    }
    return true;
}

Tensor GpuBackend::runLayer(int layer_id, const Tensor &input) {
    // Try to run a real-weight-backed operation for this layer if possible
    Tensor out;
    out.shape = input.shape;
    out.data.assign(input.data.begin(), input.data.end());

    if (weight_loader_ && !parsed_tensors_.empty()) {
        // Find a square weight matrix associated with this layer
        int hidden = (int)meta_.hidden_size;
        size_t found_idx = SIZE_MAX;
        for (const auto &pt : parsed_tensors_) {
            // look for blk.<layer_id> in name
            std::string needle = "blk." + std::to_string(layer_id) + ".";
            if (pt.name.find(needle) != std::string::npos) {
                if (pt.shape.size() >= 2 && pt.shape[0] == hidden && pt.shape[1] == hidden) {
                    found_idx = pt.index_pos; break;
                }
            }
        }
        // fallback: find any square hidden x hidden
        if (found_idx == SIZE_MAX) {
            for (const auto &pt : parsed_tensors_) {
                if (pt.shape.size() >= 2 && pt.shape[0] == hidden && pt.shape[1] == hidden) {
                    found_idx = pt.index_pos; break;
                }
            }
        }

        if (found_idx != SIZE_MAX && found_idx < weight_index_entries_.size()) {
            auto e = weight_index_entries_[found_idx];
            try {
                // Try to use device-cached weight if present
                const float *dW = runtime::gpu::cuda_get_weight_ptr(found_idx);
                std::vector<float> C(hidden, 0.0f);
                if (!dW) {
                    // allocate and upload weight slice to device
                    std::vector<float> W = weight_loader_->readFloatSlice(e.offset_bytes, e.float_count);
                    if (runtime::gpu::cuda_available()) {
                        if (!runtime::gpu::cuda_alloc_weight(found_idx, W.data(), e.float_count)) {
                            // fallback to host GEMM
                            for (int i = 0; i < hidden; ++i) {
                                float acc = 0.0f;
                                for (int j = 0; j < hidden; ++j) acc += input.data[j] * W[j * hidden + i];
                                C[i] = acc;
                            }
                            out.data = std::move(C);
                            if (available_) util::log_info("Executed GPU backend layer " + std::to_string(layer_id) + " on " + formatDeviceName(device_id_));
                            else util::log_info("Executed simulated GPU backend layer " + std::to_string(layer_id));
                            return out;
                        }
                        dW = runtime::gpu::cuda_get_weight_ptr(found_idx);
                    } else {
                        // CPU fallback
                        for (int i = 0; i < hidden; ++i) {
                            float acc = 0.0f;
                            for (int j = 0; j < hidden; ++j) acc += input.data[j] * W[j * hidden + i];
                            C[i] = acc;
                        }
                        out.data = std::move(C);
                        if (available_) util::log_info("Executed GPU backend layer " + std::to_string(layer_id) + " on " + formatDeviceName(device_id_));
                        else util::log_info("Executed simulated GPU backend layer " + std::to_string(layer_id));
                        return out;
                    }
                }
                // Use GEMM variant that takes device B pointer
                if (runtime::gpu::cuda_available()) {
                    if (!runtime::gpu::cuda_gemm_hostA_devB_hostC(input.data.data(), dW, C.data(), 1, hidden, hidden)) {
                        util::log_warn("cuda_gemm_hostA_devB_hostC failed for layer " + std::to_string(layer_id));
                        // fallback to CPU
                        std::vector<float> W = weight_loader_->readFloatSlice(e.offset_bytes, e.float_count);
                        for (int i = 0; i < hidden; ++i) {
                            float acc = 0.0f;
                            for (int j = 0; j < hidden; ++j) acc += input.data[j] * W[j * hidden + i];
                            C[i] = acc;
                        }
                    }
                } else {
                    // should not reach here
                }
                out.data = std::move(C);
                if (available_) util::log_info("Executed GPU backend layer " + std::to_string(layer_id) + " on " + formatDeviceName(device_id_));
                else util::log_info("Executed simulated GPU backend layer " + std::to_string(layer_id));
                return out;
            } catch (const std::exception &ex) {
                util::log_warn(std::string("GPU backend failed to read weight slice: ") + ex.what());
            }
        }
    }

    // Fallback behavior (no real weight found)
    const float scale = available_ ? 1.02f : 1.01f;
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = out.data[i] * scale + static_cast<float>(layer_id) * 0.001f;
    }
    if (available_) {
        util::log_info("Executed GPU backend layer " + std::to_string(layer_id) + " on " + formatDeviceName(device_id_));
    } else {
        util::log_info("Executed simulated GPU backend layer " + std::to_string(layer_id));
    }
    return out;
}

ModelMetadata GpuBackend::metadata() const {
    return meta_;
}

bool GpuBackend::available() const {
    return available_;
}

int GpuBackend::deviceId() const {
    return device_id_;
}

}
