#include "inference.h"
#include "..\\scheduler\\scheduler.h"
#include "..\\util\\logger.h"
#include <thread>
#include <string>

InferenceController::InferenceController(LayerLoader &loader, LayerCache &cache, Prefetcher &prefetcher, Scheduler *scheduler, KVManager *kv, KVPager *kvpager)
    : loader_(loader), cache_(cache), prefetcher_(prefetcher), scheduler_(scheduler), kv_(kv), kvpager_(kvpager) {}

void InferenceController::start() {
    prefetcher_.start();
    if (kvpager_) kvpager_->start();
}

void InferenceController::stop() {
    prefetcher_.stop();
    if (kvpager_) kvpager_->stop();
}

Tensor InferenceController::requestLayer(int layer_id) {
    {
        // Check cache without holding metrics atomics under mutex
        if (cache_.contains(layer_id)) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            Tensor t = cache_.get(layer_id);
            schedulePrefetch(layer_id);
            return t;
        }
    }

    // Miss: load and insert
    loader::Tensor lt = loader_.load(layer_id);
    Tensor t_native(lt.shape, DataType::FP32);
    t_native.allocateCPU();
    float* dst_native = t_native.cpu();
    for (size_t i = 0; i < lt.data.size(); ++i) dst_native[i] = lt.data[i];
    cache_.insert(layer_id, t_native);
    misses_.fetch_add(1, std::memory_order_relaxed);
    schedulePrefetch(layer_id);
    return t_native;
}

CacheMetrics InferenceController::metrics() {
    CacheMetrics s;
    s.hits = hits_.load(std::memory_order_relaxed);
    s.misses = misses_.load(std::memory_order_relaxed);
    return s;
}

void InferenceController::appendKVPage(const KVPage &p) {
    if (!kv_) return;
    // save_page will persist and keep in-memory with eviction
    kv_->save_page(p);
}

KVPage InferenceController::loadKVPage(int id) {
    if (!kv_) throw std::runtime_error("No KVManager available");
    return kv_->load_page(id);
}

void InferenceController::schedulePrefetch(int current) {
    int depth = 2;
    if (scheduler_) {
        depth = scheduler_->getPrefetchDepth(current);
        util::log_info(std::string("Scheduler decision: prefetch depth = ") + std::to_string(depth));
    }
    for (int i = 1; i <= depth; ++i) {
        int layer_id = current + i;
        // safety: only request layer if file exists
        if (loader_.exists(layer_id)) {
            prefetcher_.request(layer_id);
        } else {
            util::log_warn(std::string("Skipping prefetch for missing layer ") + std::to_string(layer_id));
        }
        if (kvpager_) {
            if (kv_ && kv_->pageExists(layer_id)) {
                kvpager_->request(layer_id);
            } else {
                // if page file doesn't exist, skip but log
                util::log_warn(std::string("Skipping KV prefetch for missing page ") + std::to_string(layer_id));
            }
        }
    }
}
