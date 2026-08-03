#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include "..\\runtime\\runtime_adapter.h"
#include "..\\runtime\\cpu_runtime.h"

int main() {
    std::filesystem::create_directories("artifacts");
    std::filesystem::path model_path = "artifacts/runtime_contract_test_model.bin";
    std::ofstream model_file(model_path, std::ios::binary);
    model_file << "stub";
    model_file.close();

    runtime::RuntimeAdapter adapter;
    if (!adapter.initialize(model_path.string())) {
        std::cerr << "failed to initialize runtime adapter" << std::endl;
        return 1;
    }

    runtime::Tensor input;
    input.data = {1.0f, 2.0f, 3.0f};
    input.shape.dims = {3};
    runtime::Tensor layer_output = adapter.executeLayer(0, input);
    std::cout << "backend metadata: " << adapter.metadata().format << "\n";
    std::cout << "output[0]=" << layer_output.data[0] << "\n";

    runtime::CpuRuntime direct_runtime;
    if (!direct_runtime.initialize(model_path.string())) {
        std::cerr << "failed to initialize CPU runtime" << std::endl;
        return 1;
    }
    runtime::Tensor direct_output = direct_runtime.executeLayer(1, input);
    std::cout << "direct runtime output[0]=" << direct_output.data[0] << "\n";
    return 0;
}