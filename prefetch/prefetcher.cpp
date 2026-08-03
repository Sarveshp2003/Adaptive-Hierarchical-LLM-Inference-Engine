#include "prefetcher.h"
#include "..\\scheduler\\scheduler.h"
#include <chrono>
#include <iostream>

#include "..\\util\\logger.h"

Prefetcher::Prefetcher(LayerLoader &loader, LayerCache &cache, Scheduler *sched)
    : loader_(loader), cache_(cache), scheduler_(sched) {}

Prefetcher::~Prefetcher() {
    stop();
}

void Prefetcher::setScheduler(Scheduler *s) {
    scheduler_ = s;
}

void Prefetcher::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    thread_ = std::thread(&Prefetcher::worker, this);
}

void Prefetcher::stop() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
}

void Prefetcher::request(int layer_id) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(layer_id);
    }
    cv_.notify_one();
}

void Prefetcher::worker() {
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
                if (!cache_.contains(id)) {
                    using clock = std::chrono::high_resolution_clock;
                    auto t0 = clock::now();
                    loader::Tensor lt = loader_.load(id);
                    Tensor t_native(lt.shape, DataType::FP32);
                    t_native.allocateCPU();
                    float* dst_native = t_native.cpu();
                    for (size_t i = 0; i < lt.data.size(); ++i) dst_native[i] = lt.data[i];
                    auto t1 = clock::now();
                    double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
                    cache_.insert(id, t_native);
                    loads.fetch_add(1, std::memory_order_relaxed);
                    util::log_info(std::string("Prefetcher loaded layer ") + std::to_string(id) + " in " + std::to_string(ms) + " ms");
                    if (scheduler_) scheduler_->observeLoadLatency(ms);
                    // small sleep to simulate IO/processing
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } catch (const std::exception &e) {
                failures.fetch_add(1, std::memory_order_relaxed);
                util::log_warn(std::string("Prefetcher failed to load layer ") + std::to_string(id) + ": " + e.what());
            }
        }
    }
    util::log_info(std::string("Prefetcher stats: requests=") + std::to_string(requests.load()) + ", loads=" + std::to_string(loads.load()) + ", failures=" + std::to_string(failures.load()));
}
