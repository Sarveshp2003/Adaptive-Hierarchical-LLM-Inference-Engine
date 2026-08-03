#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>
#include "../runtime/runtime_adapter.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: generate_benchmark <model_path> <num_tokens> <backend:cpu|gpu|auto>\n";
        return 1;
    }
    std::string model = argv[1];
    int num_tokens = std::stoi(argv[2]);
    std::string backend = argv[3];

    if (backend == "cpu") {
        _putenv_s("ADAPTIVELLM_FORCE_CPU", "1");
    } else {
        _putenv_s("ADAPTIVELLM_FORCE_CPU", "0");
    }

    runtime::RuntimeAdapter adapter;
    if (!adapter.initialize(model)) {
        std::cerr << "Failed to initialize runtime with model: " << model << "\n";
        return 2;
    }
    auto meta = adapter.metadata();
    int layers = (int)meta.num_layers;
    int hidden = (int)meta.hidden_size;
    if (layers <= 0) layers = 16; // fallback
    if (hidden <= 0) hidden = 256;

    std::cout << "Using backend for model: " << meta.name << " format=" << meta.format << " layers=" << layers << " hidden=" << hidden << "\n";

    // Prepare input tensor
    runtime::Tensor input;
    input.shape.dims = {hidden};
    input.data.assign(hidden, 0.01f);

    std::vector<double> per_token_ms;
    auto total_start = std::chrono::steady_clock::now();
    for (int t = 0; t < num_tokens; ++t) {
        auto token_start = std::chrono::steady_clock::now();
        runtime::Tensor cur = input;
        for (int l = 0; l < layers; ++l) {
            cur = adapter.executeLayer(l, cur);
        }
        auto token_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(token_end - token_start).count();
        per_token_ms.push_back(ms);
        if ((t+1) % 10 == 0) std::cout << "token=" << (t+1) << " ms=" << ms << "\n";
    }
    auto total_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_end - total_start).count();

    double sum = 0; for (double v: per_token_ms) sum += v;
    double avg = sum / per_token_ms.size();

    std::cout << "Generated " << num_tokens << " tokens in " << total_ms << " ms. avg_token_ms=" << avg << "\n";
    return 0;
}
