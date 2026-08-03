#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include "..\\cache\\kv_cache.h"

int main() {
    std::filesystem::create_directory("pages");
    // create 3 page files via KVManager (will compress on save)
    KVManager m_create(2, "pages");
    for (int i = 0; i < 3; ++i) {
        KVPage p;
        p.id = i;
        p.key = {(float)i + 0.1f, (float)i + 0.2f};
        p.value = {(float)i + 0.3f, (float)i + 0.4f};
        m_create.save_page(p);
    }

    try {
        KVManager m(2, "pages");
        auto p0 = m.load_page(0);
        std::cout << "Loaded page 0, key[0]=" << p0.key[0] << "\n";
        auto p1 = m.load_page(1);
        std::cout << "Loaded page 1, key[0]=" << p1.key[0] << "\n";
        std::cout << "In-memory size: " << m.size() << "\n";
        auto p2 = m.load_page(2);
        std::cout << "Loaded page 2, key[0]=" << p2.key[0] << "\n";
        std::cout << "In-memory size after loading 3rd: " << m.size() << "\n";
        if (m.contains(0)) {
            std::cerr << "Eviction failed: page 0 still in memory\n";
            return 2;
        }
        std::cout << "Eviction succeeded\n";
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
