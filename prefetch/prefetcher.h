#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include "..\\loader\\layer_loader.h"
#include "..\\cache\\layer_cache.h"

class Scheduler; // forward

class Prefetcher {
public:
    Prefetcher(LayerLoader &loader, LayerCache &cache, Scheduler *sched = nullptr);
    ~Prefetcher();

    // request a layer to be prefetched (thread-safe)
    void request(int layer_id);

    void start();
    void stop();

    void setScheduler(Scheduler *s);

private:
    void worker();

    LayerLoader &loader_;
    LayerCache &cache_;
    Scheduler *scheduler_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<int> q_;
    std::atomic<bool> running_ = false;
};
