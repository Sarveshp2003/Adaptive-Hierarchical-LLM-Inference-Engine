#include <iostream>
#include <fstream>
#include "..\\runtime\\runtime_adapter.h"

int main() {
    std::ofstream model_file("model.bin", std::ios::binary);
    model_file << "stub";
    model_file.close();

    runtime::RuntimeAdapter adapter;
    if (!adapter.initialize("model.bin")) {
        std::cerr << "failed to initialize runtime adapter" << std::endl;
        return 1;
    }

    runtime::Tensor input;
    input.data = {1.0f, 2.0f, 3.0f};
    input.shape.dims = {3};
    runtime::Tensor output = adapter.executeLayer(1, input);
    std::cout << "backend format: " << adapter.metadata().format << "\n";
    std::cout << "output[0]=" << output.data[0] << "\n";
    return 0;
}
