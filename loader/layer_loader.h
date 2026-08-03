#pragma once

#include <string>
#include <vector>
#include "mmap_loader.h"
#include "model_format.h"


namespace loader {
    struct Tensor {
        std::vector<float> data;
        std::vector<int> shape;
    };
}


class LayerLoader {
public:
    explicit LayerLoader(const std::string &layers_dir = "layers");

    // Load a layer by id from either the legacy binary file or the new custom model archive.
    // Returns a loader::Tensor instance.
    loader::Tensor load(int layer_id);

    // Check if layer file exists on disk
    bool exists(int layer_id) const;

    // Release resources associated with a layer (no-op for this simple impl)
    void release(int layer_id);

private:
    std::string dir_;
    std::string model_path_;
    bool is_gguf_model_ = false;
    static std::string layerPath(const std::string &dir, int id);
    static std::string archivePath(const std::string &dir);
    loader::ModelArchive loadArchive() const;
    bool archiveExists() const;
};
