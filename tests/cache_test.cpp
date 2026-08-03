#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <thread>
#include <chrono>
#include "..\\loader\\layer_loader.h"
#include "..\\cache\\layer_cache.h"
#include "..\\prefetch\\prefetcher.h"

int main() {
    std::filesystem::create_directory("layers");
    // create two tiny layer files
    {
        std::ofstream out("layers\\layer_0.bin", std::ios::binary);
        std::vector<float> v = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    }
    {
        std::ofstream out("layers\\layer_1.bin", std::ios::binary);
        std::vector<float> v = {3.0f, 4.0f};
        out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    }

    try {
        LayerLoader loader("layers");
        LayerCache cache(2);

        // load layer 0 directly (loader::Tensor) and convert to native Tensor for cache
        auto lt0 = loader.load(0);
        Tensor t0(lt0.shape, DataType::FP32);
        t0.allocateCPU();
        std::copy(lt0.data.begin(), lt0.data.end(), t0.cpu());
        cache.insert(0, t0);
        std::cout << "Cache size after insert 0: " << cache.size() << "\n";

        Prefetcher prefetch(loader, cache);
        prefetch.start();
        prefetch.request(1);

        // wait for prefetch to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        if (cache.contains(1)) {
            Tensor t1 = cache.get(1);
            // compute total element count from shape()
            size_t count = 1;
            for (auto s : t1.shape()) count *= s;
            std::cout << "Prefetched layer 1, elements: " << count << "\n";
            float *cpu = t1.cpu();
            for (size_t i = 0; i < count; ++i) std::cout << cpu[i] << " ";
            std::cout << "\n";
        } else {
            std::cerr << "Prefetch failed: layer 1 not in cache\n";
            return 2;
        }

        prefetch.stop();

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
