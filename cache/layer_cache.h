#pragma once

#include <unordered_map>
#include <list>
#include <mutex>
#include "..\\loader\\layer_loader.h"
#include "..\\native-engine\\include\\Tensor.h"

class LayerCache {
public:
    explicit LayerCache(size_t capacity = 4);

    bool contains(int id);
    void insert(int id, const Tensor &t);
    // throws if not found
    Tensor get(int id);
    size_t size();
    size_t capacity();

private:
    size_t capacity_;
    std::mutex mu_;
    std::list<int> lru_; // front = most recent
    struct Entry { Tensor t; std::list<int>::iterator it; };
    std::unordered_map<int, Entry> map_;
};
