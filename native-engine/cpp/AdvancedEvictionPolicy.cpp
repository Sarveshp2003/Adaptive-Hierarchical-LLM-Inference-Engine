#include "AdvancedEvictionPolicy.h"
#include "Logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// ============ LRU Eviction ============

int LRUEvictionPolicy::selectPageToEvict(const std::vector<EvictionMetrics>& pages) {
    if (pages.empty()) return -1;

    int lruPageId = -1;
    uint64_t minToken = UINT64_MAX;

    for (const auto& metrics : pages) {
        if (metrics.lastAccessToken < minToken) {
            minToken = metrics.lastAccessToken;
            lruPageId = metrics.pageId;
        }
    }

    return lruPageId;
}

void LRUEvictionPolicy::updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) {
    metrics.lastAccessToken = currentToken;
}

float LRUEvictionPolicy::calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) {
    return static_cast<float>(currentToken - metrics.lastAccessToken);
}

// ============ Token-Weighted Eviction ============

int TokenWeightedEvictionPolicy::selectPageToEvict(const std::vector<EvictionMetrics>& pages) {
    if (pages.empty()) return -1;

    int selectedPageId = -1;
    float maxScore = -1.0f;

    for (const auto& metrics : pages) {
        float score = calculateScore(metrics, 0);  // Current token would be passed
        if (score > maxScore) {
            maxScore = score;
            selectedPageId = metrics.pageId;
        }
    }

    return selectedPageId;
}

void TokenWeightedEvictionPolicy::updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) {
    metrics.lastAccessToken = currentToken;
}

float TokenWeightedEvictionPolicy::calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) {
    // Exponential decay: older pages get higher scores (should be evicted)
    uint64_t tokenAge = currentToken - metrics.lastAccessToken;
    return 1.0f - std::pow(decayFactor, static_cast<float>(tokenAge));
}

// ============ Attention-Based Eviction ============

int AttentionBasedEvictionPolicy::selectPageToEvict(const std::vector<EvictionMetrics>& pages) {
    if (pages.empty()) return -1;

    int selectedPageId = -1;
    float maxScore = -1.0f;

    for (const auto& metrics : pages) {
        float score = calculateScore(metrics, 0);
        if (score > maxScore) {
            maxScore = score;
            selectedPageId = metrics.pageId;
        }
    }

    return selectedPageId;
}

void AttentionBasedEvictionPolicy::updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) {
    // Called when page is accessed
    metrics.lastAccessToken = currentToken;
}

float AttentionBasedEvictionPolicy::calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) {
    // Score based on attention: low attention = high eviction score
    float attentionScore = metrics.averageAttention;
    
    // Combine with recency
    uint64_t tokenAge = currentToken - metrics.lastAccessToken;
    float recencyPenalty = std::log(tokenAge + 1.0f);

    // Eviction score: prefer low attention + old pages
    return (1.0f - attentionScore) * recencyPenalty;
}

void AttentionBasedEvictionPolicy::setAttentionWeights(int pageId, float totalAttention, float avgAttention) {
    // This would be called with attention scores from forward pass
    LOG_DEBUG("Attention weights for page " + std::to_string(pageId) + 
              ": total=" + std::to_string(totalAttention) + 
              ", avg=" + std::to_string(avgAttention));
}

// ============ Predictive Eviction ============

int PredictiveEvictionPolicy::selectPageToEvict(const std::vector<EvictionMetrics>& pages) {
    if (pages.empty()) return -1;

    int selectedPageId = -1;
    float maxScore = -1.0f;

    for (const auto& metrics : pages) {
        float score = calculateScore(metrics, 0);
        if (score > maxScore) {
            maxScore = score;
            selectedPageId = metrics.pageId;
        }
    }

    return selectedPageId;
}

void PredictiveEvictionPolicy::updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) {
    recordAccess(metrics.pageId, currentToken);
    metrics.lastAccessToken = currentToken;
}

float PredictiveEvictionPolicy::calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) {
    float nextAccessPrediction = predictNextAccess(metrics.pageId, currentToken);
    
    // Distance to predicted access: larger = should evict first
    if (nextAccessPrediction < 0) {
        return 1000.0f;  // Never accessed again
    }

    return currentToken - nextAccessPrediction;
}

float PredictiveEvictionPolicy::predictNextAccess(int pageId, uint64_t currentToken) {
    auto it = patterns.find(pageId);
    if (it == patterns.end() || it->second.accessTimes.empty()) {
        return -1.0f;  // No data
    }

    const auto& times = it->second.accessTimes;
    if (times.size() < 2) {
        return -1.0f;  // Not enough data
    }

    // Simple prediction: average interval between accesses
    uint64_t totalInterval = 0;
    for (size_t i = 1; i < times.size(); i++) {
        totalInterval += times[i] - times[i - 1];
    }
    uint64_t avgInterval = totalInterval / (times.size() - 1);

    return times.back() + avgInterval;
}

void PredictiveEvictionPolicy::recordAccess(int pageId, uint64_t token) {
    auto& pattern = patterns[pageId];
    pattern.accessTimes.push_back(token);

    // Keep only last 10 accesses to avoid memory bloat
    if (pattern.accessTimes.size() > 10) {
        pattern.accessTimes.erase(pattern.accessTimes.begin());
    }
}

// ============ Composite Manager ============

CompositeEvictionManager::CompositeEvictionManager() {
    // Initialize with LRU by default
    addPolicy(std::make_unique<LRUEvictionPolicy>(), 1.0f);
}

void CompositeEvictionManager::addPolicy(std::unique_ptr<EvictionPolicy> policy, float weight) {
    policies.push_back(std::move(policy));
    weights.push_back(weight);

    // Normalize weights
    float totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0f);
    for (auto& w : weights) {
        w /= totalWeight;
    }
}

void CompositeEvictionManager::updatePageMetrics(int pageId, const EvictionMetrics& metrics) {
    // Find or create entry
    auto it = std::find_if(pageMetrics.begin(), pageMetrics.end(),
                          [pageId](const EvictionMetrics& m) { return m.pageId == pageId; });

    if (it != pageMetrics.end()) {
        *it = metrics;
    } else {
        pageMetrics.push_back(metrics);
    }

    // Update all policies
    for (auto& policy : policies) {
        EvictionMetrics& m = pageMetrics.back();
        policy->updateMetrics(m, metrics.lastAccessToken);
    }
}

int CompositeEvictionManager::selectPageToEvict() {
    if (pageMetrics.empty()) return -1;

    if (policies.size() == 1) {
        return policies[0]->selectPageToEvict(pageMetrics);
    }

    // Composite scoring
    std::vector<float> scores(pageMetrics.size(), 0.0f);

    for (size_t p = 0; p < policies.size(); p++) {
        auto policyScores = pageMetrics;
        for (size_t i = 0; i < pageMetrics.size(); i++) {
            float score = policies[p]->calculateScore(pageMetrics[i], 0);
            scores[i] += weights[p] * score;
        }
    }

    // Find page with highest score
    int selectedPageId = -1;
    float maxScore = -1.0f;
    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            selectedPageId = pageMetrics[i].pageId;
        }
    }

    return selectedPageId;
}

void CompositeEvictionManager::switchPolicy(EvictionStrategy strategy) {
    policies.clear();
    weights.clear();

    switch (strategy) {
        case EvictionStrategy::LRU:
            addPolicy(std::make_unique<LRUEvictionPolicy>(), 1.0f);
            break;
        case EvictionStrategy::TOKEN_WEIGHTED:
            addPolicy(std::make_unique<TokenWeightedEvictionPolicy>(), 1.0f);
            break;
        case EvictionStrategy::ATTENTION_BASED:
            addPolicy(std::make_unique<AttentionBasedEvictionPolicy>(), 1.0f);
            break;
        case EvictionStrategy::PREDICTIVE:
            addPolicy(std::make_unique<PredictiveEvictionPolicy>(), 1.0f);
            break;
    }

    LOG_INFO("Switched to eviction policy: " + std::string(policies[0]->getName()));
}

const char* CompositeEvictionManager::getCurrentPolicyName() const {
    if (policies.empty()) return "NONE";
    return policies[0]->getName();
}

void CompositeEvictionManager::setPolicyWeights(const std::vector<float>& newWeights) {
    if (newWeights.size() != weights.size()) {
        LOG_ERROR("Weight count mismatch");
        return;
    }

    weights = newWeights;

    // Normalize
    float totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0f);
    for (auto& w : weights) {
        w /= totalWeight;
    }
}
