#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <string>
#include <thread>
#include "../cache/layer_cache.h"
#include "../prefetch/prefetcher.h"
#include "../loader/layer_loader.h"
#include "../runtime/runtime_adapter.h"
#include "GGUFLoader.h"

namespace {

std::filesystem::path findRepoRoot(const std::filesystem::path &start) {
    auto current = std::filesystem::absolute(start);
    while (!current.empty()) {
        if (std::filesystem::exists(current / "CMakeLists.txt")) {
            return current;
        }
        auto parent = current.parent_path();
        if (parent == current) break;
        current = parent;
    }
    return start;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path exe_path = argc > 0 ? std::filesystem::path(argv[0]) : std::filesystem::current_path();
        std::filesystem::path repo_root = findRepoRoot(exe_path.parent_path());
        std::filesystem::path model_path = argc > 1 ? std::filesystem::path(argv[1]) : repo_root / "samples" / "sample_model.adaptive";
        if (!model_path.is_absolute()) {
            model_path = repo_root / model_path;
        }

        if (!std::filesystem::exists(model_path)) {
            std::cerr << "model not found: " << model_path << std::endl;
            return 1;
        }

        LayerLoader loader(model_path.string());
        auto layer = loader.load(0);
        std::cout << "loaded layer0 with " << layer.data.size() << " weights" << std::endl;

        runtime::RuntimeAdapter adapter;
        if (!adapter.initialize(model_path.string())) {
            std::cerr << "failed to initialize runtime adapter" << std::endl;
            return 1;
        }

        auto metadata = adapter.metadata();
        std::cout << "model path: " << metadata.path << std::endl;
        std::cout << "model name: " << metadata.name << std::endl;
        std::cout << "model format: " << metadata.format << std::endl;
        std::cout << "layers: " << metadata.num_layers << " hidden: " << metadata.hidden_size << std::endl;

        if (model_path.extension().string().compare(1, 4, "gguf") == 0) {
            GGUFLoader gguf;
            if (gguf.loadMetadata(model_path.string())) {
                std::cout << "gguf metadata: general.name=" << gguf.getMetadata("general.name") << std::endl;
                std::cout << "gguf metadata: tensor count=" << gguf.getTensorNames().size() << std::endl;
                auto names = gguf.getTensorNames();
                for (size_t i = 0; i < std::min<size_t>(names.size(), 5); ++i) {
                    std::cout << "tensor[" << i << "]=" << names[i] << std::endl;
                }
            }
        }

        runtime::Tensor input;
        input.data.assign(metadata.hidden_size > 0 ? static_cast<size_t>(metadata.hidden_size) : 4, 0.25f);
        input.shape.dims = {static_cast<int64_t>(input.data.size())};
        runtime::Tensor output = adapter.executeLayer(0, input);
        runtime::Tensor output2 = adapter.executeLayer(1, output);
        std::cout << "output[0]=" << output.data[0] << std::endl;
        std::cout << "output2[0]=" << output2.data[0] << std::endl;
        std::cout << "output size=" << output.data.size() << std::endl;

        std::vector<float> prompt(input.data.begin(), input.data.end());
        auto generated = adapter.generateTokens(prompt, 3, 1.0f);
        std::cout << "generated tokens:";
        for (float v : generated) {
            std::cout << ' ' << v;
        }
        std::cout << std::endl;

        LayerCache cache(4);
        LayerLoader layer_loader(model_path.string());
        Prefetcher prefetcher(layer_loader, cache, nullptr);
        prefetcher.start();
        prefetcher.request(0);
        prefetcher.request(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        prefetcher.stop();
        std::cout << "cache size after prefetch=" << cache.size() << std::endl;
        std::cout << "runtime summary: gguf_metadata=ok inference_layers=2 generation_tokens=3 cache_prefetch=linked" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "run_sample_model exception: " << ex.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "run_sample_model unknown exception" << std::endl;
        return 3;
    }
}
