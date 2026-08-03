#pragma once

#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>

struct KVPage {
    int id;
    std::vector<float> key;   // flattened keys
    std::vector<float> value; // flattened values
};

class KVManager {
public:
    explicit KVManager(size_t capacity = 4, const std::string &pages_dir = "pages");

    // Load a page into memory (from disk if needed)
    KVPage load_page(int id);

    // Save page to disk and keep in memory
    void save_page(const KVPage &page);

    // Evict page from memory (keeps on disk)
    void evict_page(int id);

    bool contains(int id);
    bool pageExists(int id) const;
    size_t size();
    size_t capacity();

private:
    void ensure_dir();
    void persist_to_disk(const KVPage &p);
    KVPage load_from_disk(int id);

    std::string dir_;
    size_t capacity_;
    std::mutex mu_;
    std::list<int> lru_;
    std::unordered_map<int, KVPage> map_;
};
