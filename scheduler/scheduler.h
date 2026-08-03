#pragma once

#include <deque>
#include <mutex>

class LayerCache; // forward

class Scheduler {
public:
    explicit Scheduler(class LayerCache &cache, int max_depth = 4);

    // Observe a measured load latency (ms)
    void observeLoadLatency(double ms);

    // Returns prefetch depth for the given current layer id
    int getPrefetchDepth(int current);

private:
    LayerCache &cache_;
    int max_depth_;
    std::mutex mu_;
    std::deque<double> window_;
    size_t window_size_ = 16;
    double avg_latency_ms_ = 0.0;
};
