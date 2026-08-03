#include "scheduler.h"
#include "..\\cache\\layer_cache.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#undef min
#undef max
#endif

Scheduler::Scheduler(LayerCache &cache, int max_depth)
    : cache_(cache), max_depth_(max_depth) {}

void Scheduler::observeLoadLatency(double ms) {
    std::lock_guard<std::mutex> lk(mu_);
    window_.push_back(ms);
    if (window_.size() > window_size_) window_.pop_front();
    double sum = std::accumulate(window_.begin(), window_.end(), 0.0);
    avg_latency_ms_ = sum / window_.size();
}

int Scheduler::getPrefetchDepth(int /*current*/) {
    std::lock_guard<std::mutex> lk(mu_);
    double pressure = 0.0;
    double cap = 1.0;
    if (cache_.capacity() > 0) {
        pressure = static_cast<double>(cache_.size()) / static_cast<double>(cache_.capacity());
    }
    // latency factor: normalize around 50ms
    double latency_factor = std::min(2.0, avg_latency_ms_ / 50.0);

    // preference: increase depth when latency high, decrease when pressure high
    double raw = static_cast<double>(max_depth_) * latency_factor * (1.0 - pressure);
    int depth = std::max(1, std::min(max_depth_, static_cast<int>(std::round(raw))));
    return depth;
}
