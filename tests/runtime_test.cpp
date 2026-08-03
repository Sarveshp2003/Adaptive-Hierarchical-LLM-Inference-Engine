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
    // create 4 tiny layer files
    for (int i = 0; i < 4; ++i) {
        std::ofstream out("layers\\layer_" + std::to_string(i) + ".bin", std::ios::binary);
        std::vector<float> v = {(float)i*1.0f + 1.0f, (float)i*1.0f + 2.0f};
        out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    }

    try {
        LayerLoader loader("layers");
        LayerCache cache(3);
        // create scheduler and wire into prefetch and inference
        Scheduler sched(cache, 3);
        Prefetcher prefetch(loader, cache, &sched);
        KVManager kvm(4, "pages");
        KVPager kvpager(kvm, &sched);
        InferenceController ic(loader, cache, prefetch, &sched, &kvm, &kvpager);

        ic.start();

        // sequence: 0,1,2,0 (expect some hits due to prefetch)
        ic.requestLayer(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ic.requestLayer(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ic.requestLayer(2);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ic.requestLayer(0);

        auto m = ic.metrics();
        std::cout << "Metrics - hits: " << m.hits << " misses: " << m.misses << "\n";

        ic.stop();

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
