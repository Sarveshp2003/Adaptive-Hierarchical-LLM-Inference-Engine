#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "kv_cache.h"

class Scheduler; // optional

class KVPager {
public:
    explicit KVPager(KVManager &kv, Scheduler *sched = nullptr);
    ~KVPager();

    void start();
    void stop();

    // request a page to be prefetched into memory
    void request(int page_id);

private:
    void worker();

    KVManager &kv_;
    Scheduler *scheduler_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<int> q_;
    std::atomic<bool> running_{false};
};
