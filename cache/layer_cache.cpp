#include "layer_cache.h"
#include <stdexcept>

LayerCache::LayerCache(size_t capacity) : capacity_(capacity) {}

bool LayerCache::contains(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.find(id) != map_.end();
}

void LayerCache::insert(int id, const Tensor &t) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it != map_.end()) {
        // update
        it->second.t = t;
        lru_.erase(it->second.it);
        lru_.push_front(id);
        it->second.it = lru_.begin();
        return;
    }
    // evict if needed
    if (map_.size() >= capacity_) {
        int evict = lru_.back();
        lru_.pop_back();
        map_.erase(evict);
    }
    lru_.push_front(id);
    Entry e{t, lru_.begin()};
    map_.emplace(id, std::move(e));
}

Tensor LayerCache::get(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) throw std::runtime_error("LayerCache: id not found");
    // move to front
    lru_.erase(it->second.it);
    lru_.push_front(id);
    it->second.it = lru_.begin();
    return it->second.t;
}

size_t LayerCache::size() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.size();
}

size_t LayerCache::capacity() {
    return capacity_;
}
