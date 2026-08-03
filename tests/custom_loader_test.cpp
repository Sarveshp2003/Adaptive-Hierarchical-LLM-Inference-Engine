#include <iostream>
#include <filesystem>
#include <fstream>
#include "..\\loader\\layer_loader.h"

int main() {
    std::filesystem::create_directories("layers");
    std::ofstream archive("layers\\model.adaptive");
    archive << "demo-model\n";
    archive << "adaptive-custom-v1\n";
    archive << "1\n";
    archive << "2\n";
    archive << "0\n";
    archive << "layer0\n";
    archive << "2,3\n";
    archive << "1,2,3,4\n";
    archive << "1\n";
    archive << "layer1\n";
    archive << "3\n";
    archive << "5,6,7\n";
    archive.close();

    LayerLoader loader("layers");
    auto layer0 = loader.load(0);
    auto layer1 = loader.load(1);
    std::cout << "layer0 size=" << layer0.data.size() << " shape=" << layer0.shape[0] << "\n";
    std::cout << "layer1 size=" << layer1.data.size() << " shape=" << layer1.shape[0] << "\n";
    return (layer0.data.size() == 4 && layer1.data.size() == 3) ? 0 : 1;
}
