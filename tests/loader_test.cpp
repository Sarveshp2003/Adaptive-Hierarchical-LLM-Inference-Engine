#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "..\\loader\\layer_loader.h"

int main() {
    std::filesystem::create_directory("layers");
    // create a tiny layer file: 4 floats
    std::string path = "layers\\layer_0.bin";
    std::ofstream out(path, std::ios::binary);
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    out.close();

    try {
        LayerLoader loader("layers");
        auto t = loader.load(0);
        std::cout << "Loaded tensor elements: " << t.data.size() << "\n";
        for (size_t i = 0; i < t.data.size(); ++i) std::cout << t.data[i] << " ";
        std::cout << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
