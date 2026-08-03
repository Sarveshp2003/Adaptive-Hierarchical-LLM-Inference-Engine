#ifndef ADVANCED_EVICTION_POLICY_H
#define ADVANCED_EVICTION_POLICY_H

#include <vector>
#include <memory>
#include <cstdint>
#include <map>

/**
 * Advanced KV cache eviction policies.
 * 
 * Policies:
 * 1. LRU (Least Recently Used) - Base case
 * 2. Token-Weighted - Prefer recent tokens
 * 3. Attention-Based - Use attention scores
 * 4. Predictive - ML-based prediction
 */

struct EvictionMetrics {
    uint64_t lastAccessToken;
    float totalAttention;         // Sum of attention weights
    float averageAttention;       // Average attention weight
    float accessFrequency;        // Times accessed
    uint64_t creationTime;
    size_t pageSize;
    int pageId;
};

enum class EvictionStrategy {
    LRU = 0,
    TOKEN_WEIGHTED = 1,
    ATTENTION_BASED = 2,
    PREDICTIVE = 3,
};

class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    /**
     * Select page to evict
     * Returns page ID, or -1 if nothing to evict
     */
    virtual int selectPageToEvict(const std::vector<EvictionMetrics>& pages) = 0;

    /**
     * Update metrics when page is accessed
     */
    virtual void updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) = 0;

    /**
     * Calculate eviction score (higher = more likely to evict)
     */
    virtual float calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) = 0;

    virtual const char* getName() const = 0;
};

/**
 * LRU (Least Recently Used) Eviction
 */
class LRUEvictionPolicy : public EvictionPolicy {
public:
    int selectPageToEvict(const std::vector<EvictionMetrics>& pages) override;
    void updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) override;
    float calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) override;
    const char* getName() const override { return "LRU"; }
};

/**
 * Token-Weighted Eviction
 * Prefers to keep recent tokens, exponentially decays older tokens
 */
class TokenWeightedEvictionPolicy : public EvictionPolicy {
private:
    float decayFactor;  // Exponential decay rate

public:
    TokenWeightedEvictionPolicy(float decay = 0.95f) : decayFactor(decay) {}

    int selectPageToEvict(const std::vector<EvictionMetrics>& pages) override;
    void updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) override;
    float calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) override;
    const char* getName() const override { return "TokenWeighted"; }
};

/**
 * Attention-Based Eviction
 * Use attention scores to predict page importance
 * Pages with low attention weights are evicted first
 */
class AttentionBasedEvictionPolicy : public EvictionPolicy {
private:
    float attentionThreshold;

public:
    AttentionBasedEvictionPolicy(float threshold = 0.1f) : attentionThreshold(threshold) {}

    int selectPageToEvict(const std::vector<EvictionMetrics>& pages) override;
    void updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) override;
    float calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) override;
    const char* getName() const override { return "AttentionBased"; }

    void setAttentionWeights(int pageId, float totalAttention, float avgAttention);
};

/**
 * Predictive Eviction (ML-based)
 * Uses historical patterns to predict future access
 */
class PredictiveEvictionPolicy : public EvictionPolicy {
private:
    struct AccessPattern {
        std::vector<uint64_t> accessTimes;
        float predictedNextAccess;
    };

    std::map<int, AccessPattern> patterns;
    float predictionConfidence;

    /**
     * Predict when page will next be accessed
     */
    float predictNextAccess(int pageId, uint64_t currentToken);

public:
    PredictiveEvictionPolicy() : predictionConfidence(0.0f) {}

    int selectPageToEvict(const std::vector<EvictionMetrics>& pages) override;
    void updateMetrics(EvictionMetrics& metrics, uint64_t currentToken) override;
    float calculateScore(const EvictionMetrics& metrics, uint64_t currentToken) override;
    const char* getName() const override { return "Predictive"; }

    void recordAccess(int pageId, uint64_t token);
};

/**
 * Composite Eviction Manager
 * Combines multiple policies with configurable weights
 */
class CompositeEvictionManager {
private:
    std::vector<std::unique_ptr<EvictionPolicy>> policies;
    std::vector<float> weights;  // Weight for each policy
    std::vector<EvictionMetrics> pageMetrics;

public:
    CompositeEvictionManager();

    /**
     * Add policy with weight
     */
    void addPolicy(std::unique_ptr<EvictionPolicy> policy, float weight = 1.0f);

    /**
     * Update metrics for all pages
     */
    void updatePageMetrics(int pageId, const EvictionMetrics& metrics);

    /**
     * Select best page to evict using weighted combination
     */
    int selectPageToEvict();

    /**
     * Switch to specific policy
     */
    void switchPolicy(EvictionStrategy strategy);

    /**
     * Get current active policy
     */
    const char* getCurrentPolicyName() const;

    /**
     * Set policy weights
     */
    void setPolicyWeights(const std::vector<float>& newWeights);
};

#endif
