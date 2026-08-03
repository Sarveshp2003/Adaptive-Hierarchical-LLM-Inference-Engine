#include <iostream>
#include <filesystem>
#include "..\\cache\\kv_cache.h"
#include "..\\cache\\kv_pager.h"

int main() {
    std::filesystem::create_directory("pages");
    // create page files
    KVManager m(4, "pages");
    for (int i = 0; i < 4; ++i) {
        KVPage p; p.id = i; p.key = {(float)i + 0.1f}; p.value = {(float)i + 0.2f};
        m.save_page(p);
    }

    try {
        // new manager with small capacity to force disk loads
        KVManager kv(2, "pages");
        KVPager pager(kv, nullptr);
        pager.start();
        pager.request(3);
        // wait for prefetch
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!kv.contains(3)) {
            std::cerr << "KVPager failed to prefetch page 3\n";
            return 2;
        }
        pager.stop();
        std::cout << "KVPager prefetch succeeded\n";
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
