#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "..\\loader\\layer_loader.h"
#include "..\\cache\\layer_cache.h"
#include "..\\prefetch\\prefetcher.h"
#include "..\\runtime\\inference.h"
#include "..\\scheduler\\scheduler.h"

int main() {
    std::filesystem::create_directory("layers");
    // create a single layer file
    std::ofstream out("layers\\layer_0.bin", std::ios::binary);
    std::vector<float> v = {1.0f, 2.0f};
    out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    out.close();

    try {
        LayerLoader loader("layers");
        LayerCache cache(2);
        Scheduler sched(cache, 2);
        Prefetcher prefetch(loader, cache, &sched);
        KVManager kvm(4, "pages");
        KVPager kvpager(kvm, &sched);
        InferenceController ic(loader, cache, prefetch, &sched, &kvm, &kvpager);

        ic.start();

        // First request -> miss
        ic.requestLayer(0);
        // small wait
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // Second request -> should be a hit
        ic.requestLayer(0);

        auto m = ic.metrics();
        std::cout << "Metrics - hits: " << m.hits << " misses: " << m.misses << "\n";

        if (m.hits < 1) {
            std::cerr << "Expected at least 1 hit, got " << m.hits << std::endl;
            return 2;
        }
        if (m.misses < 1) {
            std::cerr << "Expected at least 1 miss, got " << m.misses << std::endl;
            return 3;
        }

        ic.stop();
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
