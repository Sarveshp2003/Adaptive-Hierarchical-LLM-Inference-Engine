#pragma once

#include <mutex>
#include <iostream>
#include <atomic>
#include "..\\loader\\layer_loader.h"
#include "..\\cache\\layer_cache.h"
#include "..\\prefetch\\prefetcher.h"

struct CacheMetrics {
    int hits = 0;
    int misses = 0;
};

#include "..\\cache\\kv_cache.h"
#include "..\\cache\\kv_pager.h"

class InferenceController {
public:
    InferenceController(LayerLoader &loader, LayerCache &cache, Prefetcher &prefetcher, class Scheduler *scheduler = nullptr, KVManager *kv = nullptr, KVPager *kvpager = nullptr);

    // Request a layer for computation; updates metrics and schedules prefetches
    Tensor requestLayer(int layer_id);

    // Return a snapshot of cache metrics
    CacheMetrics metrics();

    void start();
    void stop();

    // KV paging helpers (simple integration)
    void appendKVPage(const KVPage &p);
    KVPage loadKVPage(int id);

private:
    LayerLoader &loader_;
    LayerCache &cache_;
    Prefetcher &prefetcher_;
    class Scheduler *scheduler_ = nullptr;
    KVManager *kv_ = nullptr;
    KVPager *kvpager_ = nullptr;
    std::mutex mu_;
    std::atomic<int> hits_{0};
    std::atomic<int> misses_{0};

    // Simple scheduling rule: prefetch next N layers
    void schedulePrefetch(int current);
};
