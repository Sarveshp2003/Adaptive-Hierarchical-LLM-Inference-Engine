#include "kv_pager.h"
#include "..\\scheduler\\scheduler.h"
#include "..\\util\\logger.h"
#include <chrono>
#include <iostream>

KVPager::KVPager(KVManager &kv, Scheduler *sched) : kv_(kv), scheduler_(sched) {}

KVPager::~KVPager() {
    stop();
}

void KVPager::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    thread_ = std::thread(&KVPager::worker, this);
}

void KVPager::stop() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
}

void KVPager::request(int page_id) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(page_id);
    }
    cv_.notify_one();
}

void KVPager::worker() {
    using clock = std::chrono::high_resolution_clock;
    static std::atomic<int> requests{0}, loads{0}, failures{0};
    while (running_) {
        int id = -1;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]() { return !q_.empty() || !running_; });
            if (!running_ && q_.empty()) break;
            if (!q_.empty()) { id = q_.front(); q_.pop(); }
        }
        if (id >= 0) {
            requests.fetch_add(1, std::memory_order_relaxed);
            try {
                if (!kv_.contains(id)) {
                    auto t0 = clock::now();
                    KVPage p = kv_.load_page(id);
                    auto t1 = clock::now();
                    double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
                    loads.fetch_add(1, std::memory_order_relaxed);
                    util::log_info(std::string("KVPager loaded page ") + std::to_string(id) + " in " + std::to_string(ms) + " ms");
                    if (scheduler_) scheduler_->observeLoadLatency(ms);
                    // tiny sleep to simulate processing
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            } catch (const std::exception &e) {
                failures.fetch_add(1, std::memory_order_relaxed);
                util::log_warn(std::string("KVPager failed to load page ") + std::to_string(id) + ": " + e.what());
            }
        }
    }
    util::log_info(std::string("KVPager stats: requests=") + std::to_string(requests.load()) + ", loads=" + std::to_string(loads.load()) + ", failures=" + std::to_string(failures.load()));
}
