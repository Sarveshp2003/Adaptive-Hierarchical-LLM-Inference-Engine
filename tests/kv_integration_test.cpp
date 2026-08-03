#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "..\\loader\\layer_loader.h"
#include "..\\cache\\layer_cache.h"
#include "..\\prefetch\\prefetcher.h"
#include "..\\scheduler\\scheduler.h"
#include "..\\cache\\kv_cache.h"
#include "..\\runtime\\inference.h"

int main() {
    std::filesystem::create_directory("layers");
    std::filesystem::create_directory("pages");

    // minimal layers
    std::ofstream outL("layers\\layer_0.bin", std::ios::binary);
    std::vector<float> lv = {1.0f}; outL.write(reinterpret_cast<const char*>(lv.data()), sizeof(float));

    try {
        LayerLoader loader("layers");
        LayerCache cache(2);
        Scheduler sched(cache, 2);
        Prefetcher prefetch(loader, cache, &sched);
        KVManager kv(2, "pages");
        KVPager kvpager(kv, &sched);
        InferenceController ic(loader, cache, prefetch, &sched, &kv, &kvpager);

        // Simulate appending KV pages as context grows
        for (int i = 0; i < 4; ++i) {
            KVPage p;
            p.id = i;
            p.key = { (float)i + 0.1f };
            p.value = { (float)i + 0.2f };
            ic.appendKVPage(p);
            std::cout << "Appended KV page " << i << "\n";
        }

        // After appending 4 pages with capacity=2, only 2 should be in memory
        std::cout << "KV in-memory size: " << kv.size() << " (capacity " << kv.capacity() << ")\n";
        if (kv.size() != 2) {
            std::cerr << "Unexpected in-memory KV size\n";
            return 2;
        }

        // Try to load an evicted page (should load from disk)
        KVPage p0 = ic.loadKVPage(0);
        std::cout << "Loaded evicted page 0 key[0]=" << p0.key[0] << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
