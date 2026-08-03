#include "kv_cache.h"
#include "..\\compression\\kv_compress.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

KVManager::KVManager(size_t capacity, const std::string &pages_dir)
    : dir_(pages_dir), capacity_(capacity) {
    ensure_dir();
}

void KVManager::ensure_dir() {
    std::filesystem::create_directory(dir_);
}

std::string pagePath(const std::string &dir, int id) {
    return dir + std::string("\\page_") + std::to_string(id) + ".bin";
}

void KVManager::persist_to_disk(const KVPage &p) {
    // Write compressed (float16) key/value to disk for prototype
    std::string path = pagePath(dir_, p.id);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("KVManager: failed to open file for write: " + path);

    uint8_t compressed_flag = 1; // 1 == compressed float16
    out.write(reinterpret_cast<const char*>(&compressed_flag), sizeof(compressed_flag));

    // compress keys
    auto k16 = float32_to_float16(p.key);
    uint64_t ksz = k16.size();
    out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
    if (ksz) out.write(reinterpret_cast<const char*>(k16.data()), ksz * sizeof(uint16_t));

    // compress values
    auto v16 = float32_to_float16(p.value);
    uint64_t vsz = v16.size();
    out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
    if (vsz) out.write(reinterpret_cast<const char*>(v16.data()), vsz * sizeof(uint16_t));
}

KVPage KVManager::load_from_disk(int id) {
    std::string path = dir_ + std::string("\\page_") + std::to_string(id) + ".bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("KVManager: failed to open page file: " + path);

    uint8_t compressed_flag = 0;
    in.read(reinterpret_cast<char*>(&compressed_flag), sizeof(compressed_flag));

    KVPage p;
    p.id = id;

    if (compressed_flag == 1) {
        uint64_t ksz = 0;
        in.read(reinterpret_cast<char*>(&ksz), sizeof(ksz));
        if (ksz) {
            std::vector<uint16_t> k16(ksz);
            in.read(reinterpret_cast<char*>(k16.data()), ksz * sizeof(uint16_t));
            p.key = float16_to_float32(k16);
        }
        uint64_t vsz = 0;
        in.read(reinterpret_cast<char*>(&vsz), sizeof(vsz));
        if (vsz) {
            std::vector<uint16_t> v16(vsz);
            in.read(reinterpret_cast<char*>(v16.data()), vsz * sizeof(uint16_t));
            p.value = float16_to_float32(v16);
        }
    } else {
        // legacy: raw floats
        uint64_t ksz=0, vsz=0;
        in.read(reinterpret_cast<char*>(&ksz), sizeof(ksz));
        in.read(reinterpret_cast<char*>(&vsz), sizeof(vsz));
        if (ksz) {
            p.key.resize(ksz);
            in.read(reinterpret_cast<char*>(p.key.data()), ksz * sizeof(float));
        }
        if (vsz) {
            p.value.resize(vsz);
            in.read(reinterpret_cast<char*>(p.value.data()), vsz * sizeof(float));
        }
    }
    return p;
}

bool KVManager::pageExists(int id) const {
    std::string path = dir_ + std::string("\\page_") + std::to_string(id) + ".bin";
    return std::filesystem::exists(path);
}

KVPage KVManager::load_page(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it != map_.end()) {
        // move to front
        lru_.erase(std::find(lru_.begin(), lru_.end(), id));
        lru_.push_front(id);
        return it->second;
    }
    // load from disk
    KVPage p = load_from_disk(id);
    // insert and possibly evict
    if (map_.size() >= capacity_) {
        int ev = lru_.back(); lru_.pop_back(); map_.erase(ev);
    }
    lru_.push_front(id);
    map_.emplace(id, p);
    return p;
}

void KVManager::save_page(const KVPage &page) {
    std::lock_guard<std::mutex> lk(mu_);
    persist_to_disk(page);
    auto it = map_.find(page.id);
    if (it != map_.end()) {
        // update
        it->second = page;
        lru_.erase(std::find(lru_.begin(), lru_.end(), page.id));
        lru_.push_front(page.id);
    } else {
        if (map_.size() >= capacity_) {
            int ev = lru_.back(); lru_.pop_back(); map_.erase(ev);
        }
        lru_.push_front(page.id);
        map_.emplace(page.id, page);
    }
}

void KVManager::evict_page(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it != map_.end()) {
        lru_.erase(std::find(lru_.begin(), lru_.end(), id));
        map_.erase(it);
    }
}

bool KVManager::contains(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.find(id) != map_.end();
}

size_t KVManager::size() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.size();
}

size_t KVManager::capacity() { return capacity_; }
